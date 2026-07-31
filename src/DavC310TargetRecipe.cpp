/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "DavC310TargetRecipe.h"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <utility>

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Casting.h"

namespace ascify {

constexpr const char *DavC310TargetRecipe::TargetHeader;
constexpr const char *DavC310TargetRecipe::FunctionBinding;
constexpr const char *DavC310TargetRecipe::RecordBinding;

namespace {

using clang::ASTContext;
using clang::BinaryOperator;
using clang::CXXRecordDecl;
using clang::Decl;
using clang::DeclRefExpr;
using clang::EnumConstantDecl;
using clang::Expr;
using clang::FieldDecl;
using clang::FunctionDecl;
using clang::FunctionTemplateDecl;
using clang::IfStmt;
using clang::LangOptions;
using clang::NamedDecl;
using clang::NonTypeTemplateParmDecl;
using clang::ParmVarDecl;
using clang::QualType;
using clang::SourceLocation;
using clang::SourceManager;
using clang::Stmt;
using clang::TemplateParameterList;
using clang::TemplateTypeParmDecl;
using clang::VarDecl;

const Expr *stripExpr(const Expr *expression) {
  if (expression == nullptr)
    return nullptr;
  if (const auto *defaultArgument =
          llvm::dyn_cast<clang::CXXDefaultArgExpr>(expression))
    expression = defaultArgument->getExpr();
  return expression->IgnoreParenImpCasts();
}

bool volatileLValueToRValue(
    const clang::ImplicitCastExpr *cast) {
  return cast != nullptr &&
         cast->getCastKind() ==
             clang::CK_LValueToRValue &&
         cast->getSubExpr() != nullptr &&
         cast->getSubExpr()
             ->getType().isVolatileQualified();
}

bool constantBooleanValue(const Expr *expression,
                          ASTContext &context,
                          bool &value) {
  Expr::EvalResult evaluated;
  if (expression == nullptr ||
      !expression->EvaluateAsInt(evaluated, context) ||
      !evaluated.Val.isInt())
    return false;
  value = evaluated.Val.getInt() != 0;
  return true;
}

bool discardedOrDeferredExecution(const Stmt *statement,
                                  ASTContext &context) {
  const Stmt *current = statement;
  std::set<const Stmt *> active;
  while (current != nullptr &&
         active.insert(current).second) {
    const auto parents = context.getParents(*current);
    const Stmt *next = nullptr;
    for (const clang::DynTypedNode &parent : parents) {
      if (const auto *lambda =
              parent.get<clang::LambdaExpr>()) {
        if (lambda->getBody() == current)
          return true;
      }
      if (const auto *conditional =
              parent.get<IfStmt>()) {
        bool condition = false;
        if (constantBooleanValue(
                conditional->getCond(), context,
                condition) &&
            ((conditional->getThen() == current &&
              !condition) ||
             (conditional->getElse() == current &&
              condition)))
          return true;
      }
      if (const auto *conditional =
              parent.get<clang::ConditionalOperator>()) {
        bool condition = false;
        if (constantBooleanValue(
                conditional->getCond(), context,
                condition) &&
            ((conditional->getTrueExpr() == current &&
              !condition) ||
             (conditional->getFalseExpr() == current &&
              condition)))
          return true;
      }
      if (const auto *binary =
              parent.get<BinaryOperator>()) {
        if (binary->getRHS() == current &&
            (binary->getOpcode() == clang::BO_LAnd ||
             binary->getOpcode() == clang::BO_LOr)) {
          bool left = false;
          if (constantBooleanValue(
                  binary->getLHS(), context, left) &&
              ((binary->getOpcode() == clang::BO_LAnd &&
                !left) ||
               (binary->getOpcode() == clang::BO_LOr &&
                left)))
            return true;
        }
      }
      if (const auto *loop =
              parent.get<clang::ForStmt>()) {
        bool condition = false;
        if (loop->getBody() == current &&
            constantBooleanValue(
                loop->getCond(), context, condition) &&
            !condition)
          return true;
      }
      if (const auto *loop =
              parent.get<clang::WhileStmt>()) {
        bool condition = false;
        if (loop->getBody() == current &&
            constantBooleanValue(
                loop->getCond(), context, condition) &&
            !condition)
          return true;
      }
      if (next == nullptr)
        next = parent.get<Stmt>();
    }
    current = next;
  }
  return false;
}

bool hasIfAncestor(const Stmt *statement,
                   ASTContext &context) {
  const Stmt *current = statement;
  std::set<const Stmt *> active;
  while (current != nullptr &&
         active.insert(current).second) {
    const auto parents = context.getParents(*current);
    const Stmt *next = nullptr;
    for (const clang::DynTypedNode &parent : parents) {
      if (parent.get<IfStmt>() != nullptr)
        return true;
      if (next == nullptr)
        next = parent.get<Stmt>();
    }
    current = next;
  }
  return false;
}

const Stmt *stripAttributedStmt(const Stmt *statement) {
  while (const auto *attributed =
             llvm::dyn_cast_or_null<clang::AttributedStmt>(statement))
    statement = attributed->getSubStmt();
  return statement;
}

const Decl *directReferencedDecl(const Expr *expression) {
  expression = stripExpr(expression);
  const auto *reference =
      llvm::dyn_cast_or_null<DeclRefExpr>(expression);
  return reference == nullptr ? nullptr : reference->getDecl();
}

template<typename DeclType>
const DeclType *directReferencedAs(const Expr *expression) {
  return llvm::dyn_cast_or_null<DeclType>(
      directReferencedDecl(expression));
}

class ContainsDeclVisitor
    : public clang::RecursiveASTVisitor<ContainsDeclVisitor> {
public:
  explicit ContainsDeclVisitor(const Decl *wanted) : wanted(wanted) {}

  bool VisitDeclRefExpr(DeclRefExpr *reference) {
    if (reference->getDecl() == wanted)
      found = true;
    return !found;
  }

  bool contains(const Stmt *statement) {
    found = false;
    if (statement != nullptr)
      TraverseStmt(const_cast<Stmt *>(statement));
    return found;
  }

private:
  const Decl *wanted;
  bool found = false;
};

bool containsDecl(const Stmt *statement, const Decl *declaration) {
  if (statement == nullptr || declaration == nullptr)
    return false;
  ContainsDeclVisitor visitor(declaration);
  return visitor.contains(statement);
}

class ContainsFieldVisitor
    : public clang::RecursiveASTVisitor<ContainsFieldVisitor> {
public:
  explicit ContainsFieldVisitor(const FieldDecl *wanted)
      : wanted(wanted) {}

  bool VisitMemberExpr(clang::MemberExpr *member) {
    if (member->getMemberDecl() == wanted)
      found = true;
    return !found;
  }

  bool contains(const Stmt *statement) {
    found = false;
    if (statement != nullptr)
      TraverseStmt(const_cast<Stmt *>(statement));
    return found;
  }

private:
  const FieldDecl *wanted;
  bool found = false;
};

bool containsField(const Stmt *statement, const FieldDecl *field) {
  if (statement == nullptr || field == nullptr)
    return false;
  ContainsFieldVisitor visitor(field);
  return visitor.contains(statement);
}

bool directlyReferencesField(const Expr *expression,
                             const FieldDecl *field) {
  const auto *member =
      llvm::dyn_cast_or_null<clang::MemberExpr>(
          stripExpr(expression));
  return member != nullptr && member->getMemberDecl() == field;
}

bool isIntegralWidthAtLeast(QualType type,
                            ASTContext &context,
                            unsigned bits) {
  if (type.isNull())
    return false;
  type = type.getNonReferenceType().getUnqualifiedType();
  return type->isIntegerType() && !type.isVolatileQualified() &&
         context.getTypeSize(type) >= bits;
}

bool templateTypeParameter(QualType type,
                           unsigned expectedDepth,
                           unsigned expectedIndex,
                           bool stripPointer,
                           bool requireConstPointee,
                           bool requireMutablePointee) {
  if (type.isNull())
    return false;
  type = type.getNonReferenceType();
  if (stripPointer) {
    const auto *pointer = type->getAs<clang::PointerType>();
    if (pointer == nullptr)
      return false;
    type = pointer->getPointeeType();
    if (requireConstPointee && !type.isConstQualified())
      return false;
    if (requireMutablePointee && type.isConstQualified())
      return false;
  }
  type = type.getUnqualifiedType();
  const auto *parameter = type->getAs<clang::TemplateTypeParmType>();
  return parameter != nullptr &&
         parameter->getDepth() == expectedDepth &&
         parameter->getIndex() == expectedIndex;
}

bool templateElementType(QualType type,
                         unsigned expectedDepth,
                         unsigned expectedIndex) {
  type = type.getNonReferenceType();
  if (const auto *pointer =
          type->getAs<clang::PointerType>())
    type = pointer->getPointeeType();
  return templateTypeParameter(
      type, expectedDepth, expectedIndex,
      false, false, false);
}

bool evaluatesToPositiveInteger(const Expr *expression,
                                ASTContext &context,
                                uint64_t &value) {
  expression = stripExpr(expression);
  if (expression == nullptr)
    return false;
  Expr::EvalResult result;
  if (!expression->EvaluateAsInt(result, context) ||
      !result.Val.isInt())
    return false;
  const llvm::APSInt &integer = result.Val.getInt();
  if ((integer.isSigned() && integer.isNegative()) ||
      integer.getBitWidth() > 64)
    return false;
  value = integer.getZExtValue();
  return value != 0;
}

bool sameReferenceExpression(const Expr *lhs, const Expr *rhs) {
  lhs = stripExpr(lhs);
  rhs = stripExpr(rhs);
  if (lhs == nullptr || rhs == nullptr)
    return false;

  if (const auto *lhsReference = llvm::dyn_cast<DeclRefExpr>(lhs)) {
    const auto *rhsReference = llvm::dyn_cast<DeclRefExpr>(rhs);
    return rhsReference != nullptr &&
           lhsReference->getDecl() == rhsReference->getDecl();
  }
  if (const auto *lhsMember = llvm::dyn_cast<clang::MemberExpr>(lhs)) {
    const auto *rhsMember = llvm::dyn_cast<clang::MemberExpr>(rhs);
    return rhsMember != nullptr &&
           lhsMember->getMemberDecl() == rhsMember->getMemberDecl() &&
           sameReferenceExpression(
               lhsMember->getBase(), rhsMember->getBase());
  }
  if (const auto *lhsSubscript =
          llvm::dyn_cast<clang::ArraySubscriptExpr>(lhs)) {
    const auto *rhsSubscript =
        llvm::dyn_cast<clang::ArraySubscriptExpr>(rhs);
    return rhsSubscript != nullptr &&
           sameReferenceExpression(
               lhsSubscript->getBase(), rhsSubscript->getBase()) &&
           sameReferenceExpression(
               lhsSubscript->getIdx(), rhsSubscript->getIdx());
  }
  if (const auto *lhsBinary = llvm::dyn_cast<BinaryOperator>(lhs)) {
    const auto *rhsBinary = llvm::dyn_cast<BinaryOperator>(rhs);
    return rhsBinary != nullptr &&
           lhsBinary->getOpcode() == rhsBinary->getOpcode() &&
           sameReferenceExpression(
               lhsBinary->getLHS(), rhsBinary->getLHS()) &&
           sameReferenceExpression(
               lhsBinary->getRHS(), rhsBinary->getRHS());
  }
  return false;
}

bool isMultiplyOf(const Expr *expression,
                  const Decl *lhsDeclaration,
                  const Decl *rhsDeclaration) {
  const auto *multiply =
      llvm::dyn_cast_or_null<BinaryOperator>(stripExpr(expression));
  if (multiply == nullptr ||
      multiply->getOpcode() != clang::BO_Mul)
    return false;
  const Decl *lhs = directReferencedDecl(multiply->getLHS());
  const Decl *rhs = directReferencedDecl(multiply->getRHS());
  return (lhs == lhsDeclaration && rhs == rhsDeclaration) ||
         (lhs == rhsDeclaration && rhs == lhsDeclaration);
}

bool isAdditionOf(const Expr *expression,
                  const Decl *oneDeclaration,
                  const Expr *&other) {
  const auto *addition =
      llvm::dyn_cast_or_null<BinaryOperator>(stripExpr(expression));
  if (addition == nullptr ||
      addition->getOpcode() != clang::BO_Add)
    return false;
  if (directReferencedDecl(addition->getLHS()) == oneDeclaration) {
    other = addition->getRHS();
    return true;
  }
  if (directReferencedDecl(addition->getRHS()) == oneDeclaration) {
    other = addition->getLHS();
    return true;
  }
  return false;
}

bool matchesPackedOffset(const Expr *expression,
                         const ParmVarDecl *row,
                         const ParmVarDecl *column,
                         const FieldDecl *stride,
                         const NonTypeTemplateParmDecl *packSize) {
  const auto *division =
      llvm::dyn_cast_or_null<BinaryOperator>(stripExpr(expression));
  if (division == nullptr ||
      division->getOpcode() != clang::BO_Div ||
      directReferencedAs<NonTypeTemplateParmDecl>(
          division->getRHS()) != packSize)
    return false;

  const Expr *rowTimesStride = nullptr;
  if (!isAdditionOf(
          division->getLHS(), column, rowTimesStride))
    return false;

  const auto *multiply =
      llvm::dyn_cast_or_null<BinaryOperator>(
          stripExpr(rowTimesStride));
  if (multiply == nullptr ||
      multiply->getOpcode() != clang::BO_Mul)
    return false;
  const bool rowThenStride =
      directReferencedAs<ParmVarDecl>(multiply->getLHS()) == row &&
      directlyReferencesField(multiply->getRHS(), stride);
  const bool strideThenRow =
      directReferencedAs<ParmVarDecl>(multiply->getRHS()) == row &&
      directlyReferencesField(multiply->getLHS(), stride);
  return rowThenStride || strideThenRow;
}

bool constructorInitializesFieldFromParameter(
    const clang::CXXConstructorDecl *constructor,
    const FieldDecl *field,
    const ParmVarDecl *parameter) {
  for (const clang::CXXCtorInitializer *initializer :
       constructor->inits()) {
    if (!initializer->isMemberInitializer() ||
        initializer->getMember() != field)
      continue;
    return containsDecl(initializer->getInit(), parameter);
  }
  return false;
}

const Expr *singleParenListArgument(const Expr *expression) {
  const auto *list =
      llvm::dyn_cast_or_null<clang::ParenListExpr>(expression);
  return list != nullptr && list->getNumExprs() == 1
             ? list->getExpr(0)
             : expression;
}

bool constructorDirectlyInitializesField(
    const clang::CXXConstructorDecl *constructor,
    const FieldDecl *field,
    const ParmVarDecl *parameter) {
  for (const clang::CXXCtorInitializer *initializer :
       constructor->inits()) {
    if (!initializer->isMemberInitializer() ||
        initializer->getMember() != field)
      continue;
    return directReferencedAs<ParmVarDecl>(
               singleParenListArgument(initializer->getInit())) ==
           parameter;
  }
  return false;
}

bool constructorDirectlyMapsFields(
    const clang::CXXConstructorDecl *constructor,
    const std::vector<const FieldDecl *> &fields,
    ASTContext &context) {
  if (constructor == nullptr ||
      constructor->isCopyOrMoveConstructor() ||
      constructor->getNumParams() != fields.size())
    return false;
  std::set<const ParmVarDecl *> used;
  for (const FieldDecl *field : fields) {
    const ParmVarDecl *match = nullptr;
    for (const ParmVarDecl *parameter :
         constructor->parameters()) {
      if (!context.hasSameType(
              parameter->getType().getCanonicalType(),
              field->getType().getCanonicalType()) ||
          !constructorDirectlyInitializesField(
              constructor, field, parameter))
        continue;
      if (match != nullptr)
        return false;
      match = parameter;
    }
    if (match == nullptr || !used.insert(match).second)
      return false;
  }
  return used.size() == fields.size();
}

bool evaluatesToZero(const Expr *expression, ASTContext &context) {
  expression = stripExpr(expression);
  if (expression == nullptr)
    return false;
  Expr::EvalResult result;
  return expression->EvaluateAsInt(result, context) &&
         result.Val.isInt() && result.Val.getInt() == 0;
}

const Stmt *onlyStatement(const Stmt *statement) {
  statement = stripAttributedStmt(statement);
  const auto *compound =
      llvm::dyn_cast_or_null<clang::CompoundStmt>(statement);
  if (compound == nullptr)
    return statement;
  return compound->size() == 1 ? *compound->body_begin() : nullptr;
}

class LoopCollector
    : public clang::RecursiveASTVisitor<LoopCollector> {
public:
  bool VisitForStmt(clang::ForStmt *loop) {
    loops.push_back(loop);
    return true;
  }
  std::vector<const clang::ForStmt *> loops;
};

bool canonicalPackLoop(const clang::ForStmt *loop,
                       const NonTypeTemplateParmDecl *packSize,
                       ASTContext &context,
                       const VarDecl *&index,
                       const Stmt *&bodyStatement) {
  index = nullptr;
  bodyStatement = nullptr;
  if (loop == nullptr || loop->getInit() == nullptr ||
      loop->getCond() == nullptr || loop->getInc() == nullptr)
    return false;
  const auto *declaration =
      llvm::dyn_cast<clang::DeclStmt>(loop->getInit());
  if (declaration == nullptr || !declaration->isSingleDecl())
    return false;
  index = llvm::dyn_cast<VarDecl>(declaration->getSingleDecl());
  if (index == nullptr || !index->hasInit() ||
      !index->getType()->isIntegerType() ||
      !evaluatesToZero(index->getInit(), context))
    return false;

  const auto *condition = llvm::dyn_cast<BinaryOperator>(
      stripExpr(loop->getCond()));
  if (condition == nullptr ||
      condition->getOpcode() != clang::BO_LT ||
      directReferencedAs<VarDecl>(condition->getLHS()) != index ||
      directReferencedAs<NonTypeTemplateParmDecl>(
          condition->getRHS()) != packSize)
    return false;

  const auto *increment = llvm::dyn_cast<clang::UnaryOperator>(
      stripExpr(loop->getInc()));
  if (increment == nullptr ||
      (increment->getOpcode() != clang::UO_PreInc &&
       increment->getOpcode() != clang::UO_PostInc) ||
      directReferencedAs<VarDecl>(
          increment->getSubExpr()) != index)
    return false;

  bodyStatement = onlyStatement(loop->getBody());
  return bodyStatement != nullptr;
}

const clang::ArraySubscriptExpr *indexedBy(
    const Expr *expression, const VarDecl *index) {
  const auto *subscript =
      llvm::dyn_cast_or_null<clang::ArraySubscriptExpr>(
          stripExpr(expression));
  return subscript != nullptr &&
                 directReferencedAs<VarDecl>(
                     subscript->getIdx()) == index
             ? subscript
             : nullptr;
}

const Expr *directStaticCastOperand(const Expr *expression) {
  expression = expression == nullptr
                   ? nullptr
                   : expression->IgnoreParens();
  const auto *cast =
      llvm::dyn_cast_or_null<clang::CXXStaticCastExpr>(expression);
  return cast == nullptr ? nullptr : stripExpr(cast->getSubExpr());
}

class ReferencedLocalCollector
    : public clang::RecursiveASTVisitor<ReferencedLocalCollector> {
public:
  bool VisitDeclRefExpr(DeclRefExpr *reference) {
    if (const auto *variable =
            llvm::dyn_cast<VarDecl>(reference->getDecl())) {
      if (!llvm::isa<ParmVarDecl>(variable))
        variables.insert(variable);
    }
    return true;
  }
  std::set<const VarDecl *> variables;
};

const VarDecl *singleReferencedLocal(const Stmt *statement) {
  ReferencedLocalCollector collector;
  collector.TraverseStmt(const_cast<Stmt *>(statement));
  return collector.variables.size() == 1
             ? *collector.variables.begin()
             : nullptr;
}

class DeclReferenceCounter
    : public clang::RecursiveASTVisitor<DeclReferenceCounter> {
public:
  explicit DeclReferenceCounter(const Decl *wanted) : wanted(wanted) {}
  bool VisitDeclRefExpr(DeclRefExpr *reference) {
    if (reference->getDecl() == wanted)
      ++count;
    return true;
  }
  const Decl *wanted;
  unsigned count = 0;
};

struct PackedMemoryTransfer {
  const VarDecl *packet = nullptr;
  const VarDecl *offset = nullptr;
  const FieldDecl *packetMember = nullptr;
  const BinaryOperator *assignment = nullptr;
};

struct DirectPacketMember {
  const VarDecl *object = nullptr;
  const FieldDecl *field = nullptr;
};

const CXXRecordDecl *variableRecord(const VarDecl *variable) {
  if (variable == nullptr)
    return nullptr;
  const CXXRecordDecl *record =
      variable->getType()->getAsCXXRecordDecl();
  if (record == nullptr) {
    const auto *specialization =
        variable->getType()
            .getNonReferenceType()
            .getUnqualifiedType()
            ->getAs<clang::TemplateSpecializationType>();
    const clang::TemplateDecl *templateDeclaration =
        specialization == nullptr
            ? nullptr
            : specialization->getTemplateName()
                  .getAsTemplateDecl();
    const auto *classTemplate =
        llvm::dyn_cast_or_null<clang::ClassTemplateDecl>(
            templateDeclaration);
    if (classTemplate != nullptr)
      record = classTemplate->getTemplatedDecl();
  }
  if (record == nullptr)
    return nullptr;
  if (const CXXRecordDecl *definition = record->getDefinition())
    record = definition;
  return record;
}

DirectPacketMember directPacketMember(const Expr *expression) {
  expression = stripExpr(expression);
  if (const auto *member =
          llvm::dyn_cast_or_null<clang::MemberExpr>(expression)) {
    return {
        directReferencedAs<VarDecl>(member->getBase()),
        llvm::dyn_cast<FieldDecl>(member->getMemberDecl())};
  }
  const auto *dependent =
      llvm::dyn_cast_or_null<
          clang::CXXDependentScopeMemberExpr>(expression);
  if (dependent == nullptr)
    return {};
  const VarDecl *object =
      directReferencedAs<VarDecl>(dependent->getBase());
  const CXXRecordDecl *record = variableRecord(object);
  if (record == nullptr)
    return {};
  const clang::DeclarationName name = dependent->getMember();
  const auto lookup = record->lookup(name);
  const FieldDecl *field = nullptr;
  for (const NamedDecl *declaration : lookup) {
    const auto *candidate =
        llvm::dyn_cast<FieldDecl>(declaration);
    if (candidate == nullptr || field != nullptr)
      return {};
    field = candidate;
  }
  return {object, field};
}

const VarDecl *directMemberObject(const Expr *expression) {
  return directPacketMember(expression).object;
}

bool exactPackedMemorySide(const Expr *expression,
                           const FieldDecl *data,
                           const VarDecl *offset) {
  const auto *dereference =
      llvm::dyn_cast_or_null<clang::UnaryOperator>(
          stripExpr(expression));
  if (dereference == nullptr ||
      dereference->getOpcode() != clang::UO_Deref)
    return false;
  const auto *addition =
      llvm::dyn_cast_or_null<BinaryOperator>(
          stripExpr(dereference->getSubExpr()));
  if (addition == nullptr ||
      addition->getOpcode() != clang::BO_Add)
    return false;
  const Expr *base = nullptr;
  if (directReferencedAs<VarDecl>(addition->getLHS()) == offset)
    base = addition->getRHS();
  else if (directReferencedAs<VarDecl>(
               addition->getRHS()) == offset)
    base = addition->getLHS();
  else
    return false;

  base = base == nullptr ? nullptr : base->IgnoreParens();
  const auto *cast =
      llvm::dyn_cast_or_null<clang::ExplicitCastExpr>(base);
  if (cast == nullptr ||
      !directlyReferencesField(cast->getSubExpr(), data))
    return false;
  return true;
}

class PackedMemoryTransferVisitor
    : public clang::RecursiveASTVisitor<PackedMemoryTransferVisitor> {
public:
  PackedMemoryTransferVisitor(const FieldDecl *data, bool load)
      : data(data), load(load) {}

  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (operation->getOpcode() != clang::BO_Assign)
      return true;
    const Expr *memorySide =
        load ? operation->getRHS() : operation->getLHS();
    const Expr *packetSide =
        load ? operation->getLHS() : operation->getRHS();
    const DirectPacketMember packet =
        directPacketMember(packetSide);
    if (packet.object == nullptr || packet.field == nullptr)
      return true;
    for (const VarDecl *candidate : offsets) {
      if (!exactPackedMemorySide(memorySide, data, candidate))
        continue;
      matches.push_back(
          {packet.object, candidate, packet.field, operation});
    }
    return true;
  }

  bool VisitVarDecl(VarDecl *variable) {
    offsets.push_back(variable);
    return true;
  }

  const FieldDecl *data;
  bool load;
  std::vector<const VarDecl *> offsets;
  std::vector<PackedMemoryTransfer> matches;
};

bool directElementAssignment(const BinaryOperator *assignment,
                             const ParmVarDecl *methodData,
                             const VarDecl *packet,
                             const FieldDecl *elementField,
                             const VarDecl *index,
                             bool load) {
  if (assignment == nullptr ||
      assignment->getOpcode() != clang::BO_Assign)
    return false;
  const auto *destination =
      indexedBy(assignment->getLHS(), index);
  const Expr *castOperand =
      directStaticCastOperand(assignment->getRHS());
  const auto *source = indexedBy(castOperand, index);
  if (destination == nullptr || source == nullptr)
    return false;
  if (load) {
    const DirectPacketMember sourceMember =
        directPacketMember(source->getBase());
    return directReferencedAs<ParmVarDecl>(
               destination->getBase()) == methodData &&
           sourceMember.object == packet &&
           sourceMember.field == elementField;
  }
  const DirectPacketMember destinationMember =
      directPacketMember(destination->getBase());
  return directReferencedAs<ParmVarDecl>(
             source->getBase()) == methodData &&
         destinationMember.object == packet &&
         destinationMember.field == elementField;
}

bool matchesColumnPackOffset(
    const Expr *expression,
    const ParmVarDecl *column,
    const NonTypeTemplateParmDecl *packSize) {
  const auto *division =
      llvm::dyn_cast_or_null<BinaryOperator>(stripExpr(expression));
  return division != nullptr &&
         division->getOpcode() == clang::BO_Div &&
         directReferencedAs<ParmVarDecl>(
             division->getLHS()) == column &&
         directReferencedAs<NonTypeTemplateParmDecl>(
             division->getRHS()) == packSize;
}

bool conditionIsParameter(const Expr *condition,
                          const NonTypeTemplateParmDecl *parameter) {
  return directReferencedAs<NonTypeTemplateParmDecl>(
             condition) == parameter;
}

bool affineElementAssignment(
    const BinaryOperator *assignment,
    const ParmVarDecl *methodData,
    const VarDecl *outputPacket,
    const FieldDecl *outputElement,
    const VarDecl *weightPacket,
    const FieldDecl *weightElement,
    const VarDecl *index) {
  if (assignment == nullptr ||
      assignment->getOpcode() != clang::BO_Assign)
    return false;
  const auto *destination =
      indexedBy(assignment->getLHS(), index);
  const auto *multiply = llvm::dyn_cast_or_null<BinaryOperator>(
      stripExpr(assignment->getRHS()));
  const DirectPacketMember destinationMember =
      destination == nullptr
          ? DirectPacketMember{}
          : directPacketMember(destination->getBase());
  if (destination == nullptr || multiply == nullptr ||
      multiply->getOpcode() != clang::BO_Mul ||
      destinationMember.object != outputPacket ||
      destinationMember.field != outputElement)
    return false;

  const Expr *operands[2] = {
      multiply->getLHS(), multiply->getRHS()};
  for (unsigned castIndex = 0; castIndex != 2; ++castIndex) {
    const auto *source = indexedBy(
        directStaticCastOperand(operands[castIndex]), index);
    const auto *weight = indexedBy(
        operands[1 - castIndex], index);
    const DirectPacketMember weightMember =
        weight == nullptr
            ? DirectPacketMember{}
            : directPacketMember(weight->getBase());
    if (source != nullptr && weight != nullptr &&
        directReferencedAs<ParmVarDecl>(
            source->getBase()) == methodData &&
        weightMember.object == weightPacket &&
        weightMember.field == weightElement)
      return true;
  }
  return false;
}

bool proveDirectPackElements(
    const clang::CXXMethodDecl *method,
    const ParmVarDecl *methodData,
    const NonTypeTemplateParmDecl *packSize,
    const VarDecl *packet,
    const FieldDecl *elementField,
    bool load,
    ASTContext &context) {
  LoopCollector loops;
  loops.TraverseStmt(const_cast<Stmt *>(method->getBody()));
  loops.loops.erase(
      std::remove_if(
          loops.loops.begin(), loops.loops.end(),
          [&](const clang::ForStmt *loop) {
            return discardedOrDeferredExecution(
                loop, context);
          }),
      loops.loops.end());
  if (loops.loops.size() != 1)
    return false;
  if (discardedOrDeferredExecution(
          loops.loops.front(), context) ||
      hasIfAncestor(loops.loops.front(), context))
    return false;
  const VarDecl *index = nullptr;
  const Stmt *loopBody = nullptr;
  if (!canonicalPackLoop(
          loops.loops.front(), packSize, context, index, loopBody))
    return false;
  const auto *assignment = llvm::dyn_cast<BinaryOperator>(
      stripAttributedStmt(loopBody));
  if (!directElementAssignment(
          assignment, methodData, packet,
          elementField, index, load))
    return false;
  DeclReferenceCounter dataReferences(methodData);
  dataReferences.TraverseStmt(
      const_cast<Stmt *>(method->getBody()));
  DeclReferenceCounter packetReferences(packet);
  packetReferences.TraverseStmt(
      const_cast<Stmt *>(method->getBody()));
  return dataReferences.count == 1 &&
         packetReferences.count == 2;
}

bool proveAffinePackElements(
    const clang::CXXMethodDecl *method,
    const ParmVarDecl *methodData,
    const NonTypeTemplateParmDecl *packSize,
    const NonTypeTemplateParmDecl *affine,
    const VarDecl *outputPacket,
    const FieldDecl *outputElement,
    const VarDecl *weightPacket,
    const FieldDecl *weightElement,
    ASTContext &context) {
  LoopCollector loops;
  loops.TraverseStmt(const_cast<Stmt *>(method->getBody()));
  loops.loops.erase(
      std::remove_if(
          loops.loops.begin(), loops.loops.end(),
          [&](const clang::ForStmt *loop) {
            return discardedOrDeferredExecution(
                loop, context);
          }),
      loops.loops.end());
  if (loops.loops.size() != 1)
    return false;
  if (discardedOrDeferredExecution(
          loops.loops.front(), context) ||
      hasIfAncestor(loops.loops.front(), context))
    return false;
  const VarDecl *index = nullptr;
  const Stmt *loopBody = nullptr;
  if (!canonicalPackLoop(
          loops.loops.front(), packSize, context, index, loopBody))
    return false;
  const auto *conditional =
      llvm::dyn_cast<IfStmt>(stripAttributedStmt(loopBody));
  if (conditional == nullptr || conditional->getElse() == nullptr ||
      !conditionIsParameter(conditional->getCond(), affine))
    return false;
  const auto *affineAssignment = llvm::dyn_cast_or_null<BinaryOperator>(
      stripAttributedStmt(onlyStatement(conditional->getThen())));
  const auto *directAssignment = llvm::dyn_cast_or_null<BinaryOperator>(
      stripAttributedStmt(onlyStatement(conditional->getElse())));
  if (!affineElementAssignment(
          affineAssignment, methodData, outputPacket,
          outputElement, weightPacket,
          weightElement, index) ||
      !directElementAssignment(
          directAssignment, methodData, outputPacket,
          outputElement, index, false))
    return false;
  DeclReferenceCounter dataReferences(methodData);
  dataReferences.TraverseStmt(
      const_cast<Stmt *>(method->getBody()));
  DeclReferenceCounter outputReferences(outputPacket);
  outputReferences.TraverseStmt(
      const_cast<Stmt *>(method->getBody()));
  DeclReferenceCounter weightReferences(weightPacket);
  weightReferences.TraverseStmt(
      const_cast<Stmt *>(method->getBody()));
  return dataReferences.count == 2 &&
         outputReferences.count == 3 &&
         weightReferences.count == 2;
}

enum class AdapterKind {
  None,
  RowMajorInput,
  RowMajorOutput,
};

struct AdapterProof {
  AdapterKind kind = AdapterKind::None;
  const CXXRecordDecl *record = nullptr;
  const TemplateTypeParmDecl *storageType = nullptr;
  const TemplateTypeParmDecl *computeType = nullptr;
  const FieldDecl *data = nullptr;
  const FieldDecl *stride = nullptr;
  const FieldDecl *weight = nullptr;
  const NonTypeTemplateParmDecl *affine = nullptr;
};

bool nonVolatileTemplatePointer(QualType type,
                                unsigned templateIndex,
                                bool requireConstPointee,
                                bool requireMutablePointee) {
  if (type.isNull() || type.isVolatileQualified())
    return false;
  const auto *pointer = type->getAs<clang::PointerType>();
  if (pointer == nullptr ||
      pointer->getPointeeType().isVolatileQualified())
    return false;
  return templateTypeParameter(
      type, 0, templateIndex, true,
      requireConstPointee, requireMutablePointee);
}

bool findPackedTransfer(
    const std::vector<PackedMemoryTransfer> &matches,
    const ParmVarDecl *row,
    const ParmVarDecl *column,
    const FieldDecl *stride,
    const NonTypeTemplateParmDecl *packSize,
    PackedMemoryTransfer &result) {
  bool found = false;
  for (const PackedMemoryTransfer &match : matches) {
    if (match.offset == nullptr || !match.offset->hasInit() ||
        !match.offset->getType().isConstQualified() ||
        !matchesPackedOffset(
            match.offset->getInit(), row, column, stride, packSize))
      continue;
    if (found)
      return false;
    result = match;
    found = true;
  }
  return found;
}

bool findColumnPackedTransfer(
    const std::vector<PackedMemoryTransfer> &matches,
    const ParmVarDecl *column,
    const NonTypeTemplateParmDecl *packSize,
    PackedMemoryTransfer &result) {
  bool found = false;
  for (const PackedMemoryTransfer &match : matches) {
    if (match.offset == nullptr || !match.offset->hasInit() ||
        !match.offset->getType().isConstQualified() ||
        !matchesColumnPackOffset(
            match.offset->getInit(), column, packSize))
      continue;
    if (found)
      return false;
    result = match;
    found = true;
  }
  return found;
}

bool sizeofType(const Expr *expression,
                QualType expected,
                ASTContext &context) {
  const auto *size =
      llvm::dyn_cast_or_null<clang::UnaryExprOrTypeTraitExpr>(
          stripExpr(expression));
  return size != nullptr &&
         size->getKind() == clang::UETT_SizeOf &&
         size->isArgumentType() &&
         context.hasSameType(
             size->getArgumentType().getCanonicalType(),
             expected.getCanonicalType());
}

bool sizeTimesPack(const Expr *expression,
                   const TemplateTypeParmDecl *elementType,
                   const NonTypeTemplateParmDecl *packSize,
                   ASTContext &context) {
  const auto *multiply =
      llvm::dyn_cast_or_null<BinaryOperator>(
          stripExpr(expression));
  if (multiply == nullptr ||
      multiply->getOpcode() != clang::BO_Mul)
    return false;
  const Expr *operands[2] = {
      multiply->getLHS(), multiply->getRHS()};
  for (unsigned sizeIndex = 0; sizeIndex != 2; ++sizeIndex) {
    if (directReferencedAs<NonTypeTemplateParmDecl>(
            operands[1 - sizeIndex]) != packSize)
      continue;
    const auto *size =
        llvm::dyn_cast_or_null<
            clang::UnaryExprOrTypeTraitExpr>(
                stripExpr(operands[sizeIndex]));
    if (size == nullptr ||
        size->getKind() != clang::UETT_SizeOf ||
        !size->isArgumentType())
      continue;
    if (templateTypeParameter(
            size->getArgumentType(), elementType->getDepth(),
            elementType->getIndex(), false, false, false))
      return true;
  }
  return false;
}

bool packetHasLayoutAssertion(
    const CXXRecordDecl *record,
    const FieldDecl *storage,
    const TemplateTypeParmDecl *elementType,
    const NonTypeTemplateParmDecl *packSize,
    ASTContext &context) {
  for (const Decl *declaration : record->decls()) {
    const auto *assertion =
        llvm::dyn_cast<clang::StaticAssertDecl>(declaration);
    if (assertion == nullptr)
      continue;
    const auto *equality =
        llvm::dyn_cast_or_null<BinaryOperator>(
            stripExpr(assertion->getAssertExpr()));
    if (equality == nullptr ||
        equality->getOpcode() != clang::BO_EQ)
      continue;
    if ((sizeofType(
             equality->getLHS(), storage->getType(), context) &&
         sizeTimesPack(
             equality->getRHS(), elementType, packSize, context)) ||
        (sizeofType(
             equality->getRHS(), storage->getType(), context) &&
         sizeTimesPack(
             equality->getLHS(), elementType, packSize, context)))
      return true;
  }
  return false;
}

bool packetInstantiationMatches(
    const VarDecl *packet,
    unsigned storageTemplateIndex,
    const NonTypeTemplateParmDecl *methodPack) {
  const auto *specialization =
      packet == nullptr
          ? nullptr
          : packet->getType()->getAs<
                clang::TemplateSpecializationType>();
  if (specialization == nullptr ||
      specialization->template_arguments().size() != 2)
    return false;
  const clang::TemplateArgument &element =
      specialization->template_arguments()[0];
  const clang::TemplateArgument &width =
      specialization->template_arguments()[1];
  return element.getKind() == clang::TemplateArgument::Type &&
         templateTypeParameter(
             element.getAsType(), 0, storageTemplateIndex,
             false, false, false) &&
         width.getKind() == clang::TemplateArgument::Expression &&
         directReferencedAs<NonTypeTemplateParmDecl>(
             width.getAsExpr()) == methodPack;
}

bool provePacketOverlay(
    const VarDecl *packet,
    const FieldDecl *storage,
    const FieldDecl *element,
    unsigned storageTemplateIndex,
    const NonTypeTemplateParmDecl *methodPack,
    ASTContext &context) {
  const CXXRecordDecl *record = variableRecord(packet);
  if (record == nullptr || !record->isUnion() ||
      storage == nullptr || element == nullptr ||
      storage == element ||
      storage->getParent()->getCanonicalDecl() !=
          record->getCanonicalDecl() ||
      element->getParent()->getCanonicalDecl() !=
          record->getCanonicalDecl() ||
      !packetInstantiationMatches(
          packet, storageTemplateIndex, methodPack))
    return false;
  const auto *classTemplate =
      record->getDescribedClassTemplate();
  if (classTemplate == nullptr)
    return false;
  const TemplateParameterList *parameters =
      classTemplate->getTemplateParameters();
  if (parameters == nullptr || parameters->size() != 2)
    return false;
  const auto *elementType =
      llvm::dyn_cast<TemplateTypeParmDecl>(
          parameters->getParam(0));
  const auto *packSize =
      llvm::dyn_cast<NonTypeTemplateParmDecl>(
          parameters->getParam(1));
  if (elementType == nullptr || packSize == nullptr)
    return false;
  const auto *array =
      llvm::dyn_cast<clang::DependentSizedArrayType>(
          element->getType().getTypePtr());
  if (array == nullptr ||
      !templateTypeParameter(
          array->getElementType(), elementType->getDepth(),
          elementType->getIndex(), false, false, false) ||
      directReferencedAs<NonTypeTemplateParmDecl>(
          array->getSizeExpr()) != packSize)
    return false;
  return packetHasLayoutAssertion(
      record, storage, elementType, packSize, context);
}

const FieldDecl *packetElementField(
    const VarDecl *packet,
    const FieldDecl *storage) {
  const CXXRecordDecl *record = variableRecord(packet);
  if (record == nullptr)
    return nullptr;
  const FieldDecl *result = nullptr;
  for (const FieldDecl *field : record->fields()) {
    if (field == storage)
      continue;
    if (!llvm::isa<clang::DependentSizedArrayType>(
            field->getType().getTypePtr()))
      continue;
    if (result != nullptr)
      return nullptr;
    result = field;
  }
  return result;
}

class AffineWeightGuardVisitor
    : public clang::RecursiveASTVisitor<AffineWeightGuardVisitor> {
public:
  AffineWeightGuardVisitor(
      const NonTypeTemplateParmDecl *affine,
      const BinaryOperator *weightTransfer)
      : affine(affine), weightTransfer(weightTransfer) {}

  bool VisitIfStmt(IfStmt *conditional) {
    if (!conditionIsParameter(conditional->getCond(), affine) ||
        conditional->getElse() != nullptr)
      return true;
    const Stmt *body = onlyStatement(conditional->getThen());
    if (stripAttributedStmt(body) == weightTransfer)
      ++matches;
    return true;
  }

  unsigned matches = 0;

private:
  const NonTypeTemplateParmDecl *affine;
  const BinaryOperator *weightTransfer;
};

class AdapterControlSafetyVisitor
    : public clang::RecursiveASTVisitor<
          AdapterControlSafetyVisitor> {
public:
  AdapterControlSafetyVisitor(
      ASTContext &context,
      const NonTypeTemplateParmDecl *affine)
      : context(context), affine(affine) {}

  bool VisitLambdaExpr(clang::LambdaExpr *) {
    safe = false;
    return false;
  }

  bool VisitConditionalOperator(
      clang::ConditionalOperator *) {
    safe = false;
    return false;
  }

  bool VisitSwitchStmt(clang::SwitchStmt *) {
    safe = false;
    return false;
  }

  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (operation->getOpcode() == clang::BO_LAnd ||
        operation->getOpcode() == clang::BO_LOr) {
      safe = false;
      return false;
    }
    return true;
  }

  bool VisitIfStmt(IfStmt *statement) {
    if (affine == nullptr ||
        !conditionIsParameter(
            statement->getCond(), affine)) {
      safe = false;
      return false;
    }
    return true;
  }

  bool VisitForStmt(clang::ForStmt *statement) {
    bool condition = true;
    if (statement->getCond() != nullptr &&
        constantBooleanValue(
            statement->getCond(), context, condition) &&
        !condition) {
      safe = false;
      return false;
    }
    return true;
  }

  bool VisitWhileStmt(clang::WhileStmt *statement) {
    safe = false;
    return false;
  }

  bool VisitDoStmt(clang::DoStmt *) {
    safe = false;
    return false;
  }

  ASTContext &context;
  const NonTypeTemplateParmDecl *affine;
  bool safe = true;
};

class AdapterEffectVisitor
    : public clang::RecursiveASTVisitor<
          AdapterEffectVisitor> {
public:
  AdapterEffectVisitor(
      std::set<const Stmt *> allowedAssignments,
      const Expr *allowedIncrement,
      ASTContext &context)
      : allowedAssignments(
            std::move(allowedAssignments)),
        allowedIncrement(stripExpr(allowedIncrement)),
        context(context) {}

  bool VisitVarDecl(VarDecl *variable) {
    if ((variable->isStaticLocal() &&
         !variable->hasAttr<
             clang::CUDASharedAttr>()) ||
        variable->getType().isVolatileQualified() ||
        variable->hasAttr<clang::CleanupAttr>())
      safe = false;
    QualType element =
        context.getBaseElementType(variable->getType());
    const CXXRecordDecl *record =
        element.isNull()
            ? nullptr
            : element->getAsCXXRecordDecl();
    if (record != nullptr &&
        !record->hasTrivialDestructor())
      safe = false;
    return safe;
  }

  bool VisitDeclRefExpr(DeclRefExpr *reference) {
    const auto *variable =
        llvm::dyn_cast<VarDecl>(
            reference->getDecl());
    if (variable != nullptr &&
        variable->getType().isVolatileQualified())
      safe = false;
    return safe;
  }

  bool VisitImplicitCastExpr(
      clang::ImplicitCastExpr *cast) {
    if (volatileLValueToRValue(cast))
      safe = false;
    return safe;
  }

  bool VisitAtomicExpr(clang::AtomicExpr *) {
    safe = false;
    return false;
  }

  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (operation->isAssignmentOp() &&
        allowedAssignments.count(operation) == 0)
      safe = false;
    return safe;
  }

  bool VisitUnaryOperator(
      clang::UnaryOperator *operation) {
    switch (operation->getOpcode()) {
    case clang::UO_PreInc:
    case clang::UO_PostInc:
    case clang::UO_PreDec:
    case clang::UO_PostDec:
      if (stripExpr(operation) != allowedIncrement)
        safe = false;
      break;
    default:
      break;
    }
    return safe;
  }

  bool VisitCallExpr(clang::CallExpr *) {
    safe = false;
    return false;
  }

  bool VisitCXXConstructExpr(
      clang::CXXConstructExpr *construction) {
    const clang::CXXConstructorDecl *constructor =
        construction->getConstructor();
    const auto *body =
        constructor == nullptr || !constructor->hasBody()
            ? nullptr
            : llvm::dyn_cast<clang::CompoundStmt>(
                  constructor->getBody());
    const CXXRecordDecl *record =
        constructor == nullptr
            ? nullptr
            : constructor->getParent();
    const clang::CXXDestructorDecl *destructor =
        record == nullptr ? nullptr
                          : record->getDestructor();
    const auto *destructorBody =
        destructor == nullptr || !destructor->hasBody()
            ? nullptr
            : llvm::dyn_cast<clang::CompoundStmt>(
                  destructor->getBody());
    const bool emptyConstructor =
        constructor != nullptr &&
        record != nullptr && record->isUnion() &&
        construction->getNumArgs() == 0 &&
        (constructor->isDefaulted() ||
         (body != nullptr && body->body_empty()));
    const bool emptyDestructor =
        destructor == nullptr ||
        destructor->isDefaulted() ||
        (destructorBody != nullptr &&
         destructorBody->body_empty());
    if (!emptyConstructor || !emptyDestructor)
      safe = false;
    return safe;
  }

  bool VisitReturnStmt(clang::ReturnStmt *) {
    safe = false;
    return false;
  }

  bool VisitGotoStmt(clang::GotoStmt *) {
    safe = false;
    return false;
  }

  bool VisitGCCAsmStmt(clang::GCCAsmStmt *) {
    safe = false;
    return false;
  }

  bool VisitMSAsmStmt(clang::MSAsmStmt *) {
    safe = false;
    return false;
  }

  bool VisitCXXThrowExpr(clang::CXXThrowExpr *) {
    safe = false;
    return false;
  }

  bool VisitCXXNewExpr(clang::CXXNewExpr *) {
    safe = false;
    return false;
  }

  bool VisitCXXDeleteExpr(clang::CXXDeleteExpr *) {
    safe = false;
    return false;
  }

  std::set<const Stmt *> allowedAssignments;
  const Expr *allowedIncrement;
  ASTContext &context;
  bool safe = true;
};

bool adapterEffectsExact(
    const clang::CXXMethodDecl *method,
    const NonTypeTemplateParmDecl *packSize,
    const NonTypeTemplateParmDecl *affine,
    const BinaryOperator *outputTransfer,
    const BinaryOperator *weightTransfer,
    ASTContext &context) {
  LoopCollector loops;
  loops.TraverseStmt(
      const_cast<Stmt *>(method->getBody()));
  if (loops.loops.size() != 1 ||
      hasIfAncestor(loops.loops.front(), context))
    return false;
  const VarDecl *index = nullptr;
  const Stmt *loopBody = nullptr;
  if (!canonicalPackLoop(
          loops.loops.front(), packSize, context,
          index, loopBody))
    return false;
  std::set<const Stmt *> allowed = {outputTransfer};
  if (weightTransfer != nullptr)
    allowed.insert(weightTransfer);
  if (affine == nullptr) {
    const auto *assignment =
        llvm::dyn_cast<BinaryOperator>(
            stripAttributedStmt(loopBody));
    if (assignment == nullptr)
      return false;
    allowed.insert(assignment);
  } else {
    const auto *conditional =
        llvm::dyn_cast<IfStmt>(
            stripAttributedStmt(loopBody));
    if (conditional == nullptr ||
        !conditionIsParameter(
            conditional->getCond(), affine))
      return false;
    const auto *affineAssignment =
        llvm::dyn_cast_or_null<BinaryOperator>(
            stripAttributedStmt(
                onlyStatement(
                    conditional->getThen())));
    const auto *directAssignment =
        llvm::dyn_cast_or_null<BinaryOperator>(
            stripAttributedStmt(
                onlyStatement(
                    conditional->getElse())));
    if (affineAssignment == nullptr ||
        directAssignment == nullptr)
      return false;
    allowed.insert(affineAssignment);
    allowed.insert(directAssignment);
  }
  AdapterEffectVisitor effects(
      std::move(allowed),
      loops.loops.front()->getInc(), context);
  effects.TraverseStmt(
      const_cast<Stmt *>(method->getBody()));
  return effects.safe;
}

AdapterProof proveDirectAdapter(const CXXRecordDecl *record,
                                ASTContext &context,
                                SourceManager &sourceManager) {
  AdapterProof proof;
  if (record == nullptr ||
      !record->isThisDeclarationADefinition() ||
      record->isImplicit() || record->isUnion() ||
      record->getNumBases() != 0 ||
      record->getLocation().isInvalid() ||
      record->getLocation().isMacroID() ||
      !sourceManager.isWrittenInMainFile(record->getLocation()))
    return proof;

  const auto *classTemplate = record->getDescribedClassTemplate();
  if (classTemplate == nullptr)
    return proof;
  const TemplateParameterList *parameters =
      classTemplate->getTemplateParameters();
  if (parameters == nullptr ||
      (parameters->size() != 2 && parameters->size() != 3))
    return proof;
  const auto *firstType = llvm::dyn_cast<TemplateTypeParmDecl>(
      parameters->getParam(0));
  const auto *secondType = llvm::dyn_cast<TemplateTypeParmDecl>(
      parameters->getParam(1));
  if (firstType == nullptr || secondType == nullptr ||
      firstType->isParameterPack() || secondType->isParameterPack() ||
      firstType->getIdentifier() == nullptr ||
      secondType->getIdentifier() == nullptr)
    return proof;
  const auto *affine =
      parameters->size() == 3
          ? llvm::dyn_cast<NonTypeTemplateParmDecl>(
                parameters->getParam(2))
          : nullptr;
  if (parameters->size() == 3 &&
      (affine == nullptr || affine->isParameterPack() ||
       !affine->getType()
            .getNonReferenceType()
            .getUnqualifiedType()->isBooleanType() ||
       affine->getIdentifier() == nullptr))
    return proof;

  static const char *const markerNames[] = {
      "ascify_target_direct_load_tag",
      "ascify_target_direct_store_tag",
      "ascify_target_adapter_owner_type",
      "ascify_target_storage_type",
      "ascify_target_compute_type",
      "ascify_target_store_is_affine",
      "ascify_target_data",
      "ascify_target_row_stride",
      "ascify_target_weight",
  };
  for (const char *name : markerNames) {
    if (!record->lookup(&context.Idents.get(name)).empty())
      return proof;
  }

  std::vector<const FieldDecl *> fields;
  for (const FieldDecl *field : record->fields()) {
    if (!field->isUnnamedBitField())
      fields.push_back(field);
  }
  if (fields.size() != (affine == nullptr ? 2 : 3))
    return proof;

  const FieldDecl *inputPointer = nullptr;
  const FieldDecl *outputPointer = nullptr;
  const FieldDecl *weightPointer = nullptr;
  const FieldDecl *strideField = nullptr;
  for (const FieldDecl *field : fields) {
    if (field->getType().isVolatileQualified())
      return proof;
    if (nonVolatileTemplatePointer(
            field->getType(), 0, true, false)) {
      if (inputPointer != nullptr)
        return proof;
      inputPointer = field;
    } else if (nonVolatileTemplatePointer(
                   field->getType(), 1, false, true)) {
      if (outputPointer != nullptr)
        return proof;
      outputPointer = field;
    } else if (nonVolatileTemplatePointer(
                   field->getType(), 1, true, false)) {
      if (weightPointer != nullptr)
        return proof;
      weightPointer = field;
    } else if (isIntegralWidthAtLeast(
                   field->getType(), context, 32) &&
               strideField == nullptr) {
      strideField = field;
    } else {
      return proof;
    }
  }
  const bool loadFields =
      affine == nullptr && inputPointer != nullptr &&
      outputPointer == nullptr && weightPointer == nullptr;
  const bool directStoreFields =
      affine == nullptr && inputPointer == nullptr &&
      outputPointer != nullptr && weightPointer == nullptr;
  const bool affineStoreFields =
      affine != nullptr && inputPointer == nullptr &&
      outputPointer != nullptr && weightPointer != nullptr;
  if ((!loadFields && !directStoreFields && !affineStoreFields) ||
      strideField == nullptr ||
      strideField->getIdentifier() == nullptr)
    return proof;
  const FieldDecl *pointerField =
      loadFields ? inputPointer : outputPointer;
  if (pointerField->getIdentifier() == nullptr ||
      (weightPointer != nullptr &&
       weightPointer->getIdentifier() == nullptr))
    return proof;

  const clang::CXXMethodDecl *adapterMethod = nullptr;
  const NonTypeTemplateParmDecl *packSize = nullptr;
  bool isLoad = false;
  unsigned deviceMethodTemplates = 0;
  for (const Decl *member : record->decls()) {
    const auto *methodTemplate =
        llvm::dyn_cast<FunctionTemplateDecl>(member);
    const auto *method =
        methodTemplate == nullptr
            ? nullptr
            : llvm::dyn_cast<clang::CXXMethodDecl>(
                  methodTemplate->getTemplatedDecl());
    if (method == nullptr || !method->hasBody() ||
        !method->hasAttr<clang::CUDADeviceAttr>())
      continue;
    ++deviceMethodTemplates;
    if (
        !method->getReturnType()->isVoidType() ||
        method->getNumParams() != 3)
      continue;
    const TemplateParameterList *methodParameters =
        methodTemplate->getTemplateParameters();
    if (methodParameters == nullptr || methodParameters->size() != 1)
      continue;
    const auto *candidatePack =
        llvm::dyn_cast<NonTypeTemplateParmDecl>(
            methodParameters->getParam(0));
    if (candidatePack == nullptr ||
        candidatePack->isParameterPack() ||
        !candidatePack->getType()->isIntegerType())
      continue;
    if (!isIntegralWidthAtLeast(
            method->getParamDecl(1)->getType(), context, 32) ||
        !isIntegralWidthAtLeast(
            method->getParamDecl(2)->getType(), context, 32) ||
        method->getParamDecl(0)->getType().isVolatileQualified() ||
        method->getParamDecl(0)->getType()
            ->getPointeeType().isVolatileQualified())
      continue;

    const QualType methodDataType =
        method->getParamDecl(0)->getType();
    const bool loadShape =
        loadFields &&
        nonVolatileTemplatePointer(
            methodDataType, 1, false, true) &&
        method->isConst();
    const bool storeShape =
        !loadFields &&
        nonVolatileTemplatePointer(
            methodDataType, 0, true, false) &&
        !method->isConst();
    if (!loadShape && !storeShape)
      continue;
    if (adapterMethod != nullptr)
      return proof;
    adapterMethod = method;
    packSize = candidatePack;
    isLoad = loadShape;
  }
  if (deviceMethodTemplates != 1 ||
      adapterMethod == nullptr || packSize == nullptr)
    return proof;
  AdapterControlSafetyVisitor controlSafety(
      context, affine);
  controlSafety.TraverseStmt(
      const_cast<Stmt *>(adapterMethod->getBody()));
  if (!controlSafety.safe)
    return proof;

  const clang::CXXConstructorDecl *matchingConstructor = nullptr;
  for (const clang::CXXConstructorDecl *constructor :
       record->ctors()) {
    if (!constructorDirectlyMapsFields(
            constructor, fields, context))
      continue;
    if (matchingConstructor != nullptr)
      return proof;
    matchingConstructor = constructor;
  }
  if (matchingConstructor == nullptr)
    return proof;

  PackedMemoryTransferVisitor transfer(pointerField, isLoad);
  transfer.TraverseStmt(
      const_cast<Stmt *>(adapterMethod->getBody()));
  transfer.matches.erase(
      std::remove_if(
          transfer.matches.begin(), transfer.matches.end(),
          [&](const PackedMemoryTransfer &match) {
            return discardedOrDeferredExecution(
                match.assignment, context);
          }),
      transfer.matches.end());
  PackedMemoryTransfer outputTransfer;
  if (!findPackedTransfer(
          transfer.matches,
          adapterMethod->getParamDecl(1),
          adapterMethod->getParamDecl(2),
          strideField, packSize, outputTransfer))
    return proof;
  if (hasIfAncestor(
          outputTransfer.assignment, context))
    return proof;
  DeclReferenceCounter outputOffsetReferences(
      outputTransfer.offset);
  outputOffsetReferences.TraverseStmt(
      const_cast<Stmt *>(adapterMethod->getBody()));
  if (outputOffsetReferences.count != 1)
    return proof;
  const FieldDecl *outputElement =
      packetElementField(
          outputTransfer.packet,
          outputTransfer.packetMember);
  const unsigned storageTemplateIndex =
      isLoad ? 0 : 1;
  if (!provePacketOverlay(
          outputTransfer.packet,
          outputTransfer.packetMember,
          outputElement, storageTemplateIndex,
          packSize, context))
    return proof;

  const BinaryOperator *provenWeightTransfer = nullptr;
  if (affine == nullptr) {
    if (!proveDirectPackElements(
            adapterMethod, adapterMethod->getParamDecl(0),
            packSize, outputTransfer.packet,
            outputElement, isLoad, context))
      return proof;
  } else {
    PackedMemoryTransferVisitor weightTransfer(
        weightPointer, true);
    weightTransfer.TraverseStmt(
        const_cast<Stmt *>(adapterMethod->getBody()));
    weightTransfer.matches.erase(
        std::remove_if(
            weightTransfer.matches.begin(),
            weightTransfer.matches.end(),
            [&](const PackedMemoryTransfer &match) {
              return discardedOrDeferredExecution(
                  match.assignment, context);
            }),
        weightTransfer.matches.end());
    PackedMemoryTransfer weightPackedTransfer;
    if (!findColumnPackedTransfer(
            weightTransfer.matches,
            adapterMethod->getParamDecl(2), packSize,
            weightPackedTransfer))
      return proof;
    provenWeightTransfer =
        weightPackedTransfer.assignment;
    DeclReferenceCounter weightOffsetReferences(
        weightPackedTransfer.offset);
    weightOffsetReferences.TraverseStmt(
        const_cast<Stmt *>(adapterMethod->getBody()));
    if (weightOffsetReferences.count != 1)
      return proof;
    const FieldDecl *weightElement =
        packetElementField(
            weightPackedTransfer.packet,
            weightPackedTransfer.packetMember);
    if (!provePacketOverlay(
            weightPackedTransfer.packet,
            weightPackedTransfer.packetMember,
            weightElement, 1, packSize, context))
      return proof;
    AffineWeightGuardVisitor weightGuard(
        affine, weightPackedTransfer.assignment);
    weightGuard.TraverseStmt(
        const_cast<Stmt *>(adapterMethod->getBody()));
    if (weightGuard.matches != 1 ||
        !proveAffinePackElements(
            adapterMethod, adapterMethod->getParamDecl(0),
            packSize, affine,
            outputTransfer.packet, outputElement,
            weightPackedTransfer.packet, weightElement,
            context))
      return proof;
  }
  if (!adapterEffectsExact(
          adapterMethod, packSize, affine,
          outputTransfer.assignment,
          provenWeightTransfer, context))
    return proof;

  proof.kind =
      isLoad ? AdapterKind::RowMajorInput
             : AdapterKind::RowMajorOutput;
  proof.record = record;
  proof.storageType = isLoad ? firstType : secondType;
  proof.computeType = isLoad ? secondType : firstType;
  proof.data = pointerField;
  proof.stride = strideField;
  proof.weight = weightPointer;
  proof.affine = affine;
  return proof;
}

std::string adapterMarkerText(const AdapterProof &proof) {
  const bool load = proof.kind == AdapterKind::RowMajorInput;
  std::string text =
      "\n public:\n"
      "  using ";
  text += load ? "ascify_target_direct_load_tag"
               : "ascify_target_direct_store_tag";
  text += " = void;\n"
          "  using ascify_target_adapter_owner_type = ";
  text += proof.record->getNameAsString();
  text += ";\n"
          "  using ascify_target_storage_type = ";
  text += proof.storageType->getNameAsString();
  text += ";\n"
          "  using ascify_target_compute_type = ";
  text += proof.computeType->getNameAsString();
  text += ";\n"
          "";
  if (!load) {
    text +=
        "  static constexpr bool "
        "ascify_target_store_is_affine = ";
    text += proof.affine == nullptr
                ? "false"
                : proof.affine->getNameAsString();
    text += ";\n";
  }
  text +=
          "  auto ascify_target_data() const -> decltype(";
  text += proof.data->getNameAsString();
  text += ") { return ";
  text += proof.data->getNameAsString();
  text += "; }\n"
          "  auto ascify_target_row_stride() const -> decltype(";
  text += proof.stride->getNameAsString();
  text += ") { return ";
  text += proof.stride->getNameAsString();
  text += "; }\n";
  if (proof.weight != nullptr) {
    text +=
        "  auto ascify_target_weight() const -> decltype(";
    text += proof.weight->getNameAsString();
    text += ") { return ";
    text += proof.weight->getNameAsString();
    text += "; }\n";
  }
  return text;
}

enum class PrimitiveKind {
  None,
  Exp,
  Divide,
  Rsqrt,
};

bool primitiveSignature(const FunctionDecl *function,
                        PrimitiveKind kind,
                        ASTContext &context) {
  const unsigned arity =
      kind == PrimitiveKind::Divide ? 2 : 1;
  if (function == nullptr || kind == PrimitiveKind::None ||
      function->getNumParams() != arity)
    return false;
  const QualType result =
      function->getReturnType()
          .getNonReferenceType()
          .getUnqualifiedType();
  if (!result->isRealFloatingType())
    return false;
  for (const ParmVarDecl *parameter : function->parameters()) {
    const QualType parameterType =
        parameter->getType()
            .getNonReferenceType()
            .getUnqualifiedType();
    if (!parameterType->isRealFloatingType() ||
        !context.hasSameType(
            result.getCanonicalType(),
            parameterType.getCanonicalType()))
      return false;
  }
  return true;
}

PrimitiveKind annotatedPrimitive(const FunctionDecl *function) {
  if (function == nullptr)
    return PrimitiveKind::None;
  PrimitiveKind result = PrimitiveKind::None;
  for (const clang::AnnotateAttr *annotation :
       function->specific_attrs<clang::AnnotateAttr>()) {
    PrimitiveKind candidate = PrimitiveKind::None;
    if (annotation->getAnnotation() == "ascify.semantic.exp")
      candidate = PrimitiveKind::Exp;
    else if (annotation->getAnnotation() ==
             "ascify.semantic.divide")
      candidate = PrimitiveKind::Divide;
    else if (annotation->getAnnotation() ==
             "ascify.semantic.rsqrt")
      candidate = PrimitiveKind::Rsqrt;
    if (candidate == PrimitiveKind::None)
      continue;
    if (result != PrimitiveKind::None && result != candidate)
      return PrimitiveKind::None;
    result = candidate;
  }
  return result;
}

PrimitiveKind standardPrimitive(const FunctionDecl *function,
                                SourceManager &sourceManager) {
  if (function == nullptr || function->getIdentifier() == nullptr ||
      !sourceManager.isInSystemHeader(
          sourceManager.getExpansionLoc(function->getLocation())))
    return PrimitiveKind::None;
  const llvm::StringRef name = function->getName();
  if (name == "exp" || name == "expf" || name == "__expf")
    return PrimitiveKind::Exp;
  if (name == "rsqrt" || name == "rsqrtf" ||
      name == "__frsqrt_rn")
    return PrimitiveKind::Rsqrt;
  if (name == "__fdividef")
    return PrimitiveKind::Divide;
  return PrimitiveKind::None;
}

const FunctionDecl *functionFromNamedDecl(
    const NamedDecl *declaration) {
  if (const auto *function =
          llvm::dyn_cast_or_null<FunctionDecl>(declaration))
    return function;
  if (const auto *functionTemplate =
          llvm::dyn_cast_or_null<FunctionTemplateDecl>(declaration))
    return functionTemplate->getTemplatedDecl();
  return nullptr;
}

class PrimitiveRegistry {
public:
  PrimitiveRegistry(
      ASTContext &context,
      SourceManager &sourceManager,
      const std::vector<const FunctionDecl *> &functions)
      : context(context), sourceManager(sourceManager) {
    for (const FunctionDecl *function : functions) {
      if (function == nullptr)
        continue;
      const FunctionTemplateDecl *primary =
          function->getPrimaryTemplate();
      if (primary == nullptr)
        continue;
      const PrimitiveKind kind = classify(function);
      if (kind == PrimitiveKind::None)
        continue;
      const FunctionDecl *key =
          primary->getTemplatedDecl()->getCanonicalDecl();
      auto inserted = templateKinds.emplace(key, kind);
      if (!inserted.second && inserted.first->second != kind)
        inserted.first->second = PrimitiveKind::None;
    }
  }

  PrimitiveKind classify(const FunctionDecl *function) {
    if (function == nullptr)
      return PrimitiveKind::None;
    const FunctionDecl *key = function->getCanonicalDecl();
    const auto cached = functionKinds.find(key);
    if (cached != functionKinds.end())
      return cached->second;
    if (!active.insert(key).second)
      return PrimitiveKind::None;

    PrimitiveKind result = annotatedPrimitive(function);
    if (result != PrimitiveKind::None &&
        !primitiveSignature(function, result, context))
      result = PrimitiveKind::None;

    if (result == PrimitiveKind::None) {
      result = standardPrimitive(function, sourceManager);
      if (result != PrimitiveKind::None &&
          !primitiveSignature(function, result, context))
        result = PrimitiveKind::None;
    }

    const FunctionDecl *definition = nullptr;
    if (result == PrimitiveKind::None &&
        function->hasBody(definition) &&
        definition != nullptr &&
        definition->hasAttr<clang::CUDADeviceAttr>()) {
      const auto *returned =
          llvm::dyn_cast_or_null<clang::ReturnStmt>(
              stripAttributedStmt(
                  onlyStatement(definition->getBody())));
      if (returned != nullptr) {
        const Expr *value = stripExpr(returned->getRetValue());
        if (const auto *division =
                llvm::dyn_cast_or_null<BinaryOperator>(value)) {
          if (division->getOpcode() == clang::BO_Div &&
              definition->getNumParams() == 2 &&
              directReferencedAs<ParmVarDecl>(
                  division->getLHS()) ==
                  definition->getParamDecl(0) &&
              directReferencedAs<ParmVarDecl>(
                  division->getRHS()) ==
                  definition->getParamDecl(1))
            result = PrimitiveKind::Divide;
        } else if (const auto *call =
                       llvm::dyn_cast_or_null<clang::CallExpr>(
                           value)) {
          const PrimitiveKind called = classifyCall(call);
          const unsigned arity =
              called == PrimitiveKind::Divide ? 2 : 1;
          if (called != PrimitiveKind::None &&
              call->getNumArgs() == arity &&
              definition->getNumParams() == arity) {
            result = called;
            for (unsigned index = 0; index < arity; ++index) {
              if (directReferencedAs<ParmVarDecl>(
                      call->getArg(index)) !=
                  definition->getParamDecl(index)) {
                result = PrimitiveKind::None;
                break;
              }
            }
          }
        }
      }
      if (result != PrimitiveKind::None &&
          !primitiveSignature(definition, result, context))
        result = PrimitiveKind::None;
    }

    active.erase(key);
    functionKinds[key] = result;
    return result;
  }

  PrimitiveKind classifyCall(
      const clang::CallExpr *call,
      bool handledFundamentalOnly = false) {
    if (call == nullptr)
      return PrimitiveKind::None;
    if (const FunctionDecl *direct = call->getDirectCallee())
      return classify(direct);

    const Expr *callee = stripExpr(call->getCallee());
    if (const auto *reference =
            llvm::dyn_cast_or_null<DeclRefExpr>(callee)) {
      if (const FunctionDecl *function =
              functionFromNamedDecl(reference->getDecl()))
        return classifyFunctionOrTemplate(function);
    }
    const auto *unresolved =
        llvm::dyn_cast_or_null<clang::UnresolvedLookupExpr>(callee);
    if (unresolved == nullptr ||
        (unresolved->requiresADL() &&
         !handledFundamentalOnly))
      return PrimitiveKind::None;
    PrimitiveKind result = PrimitiveKind::None;
    bool sawCandidate = false;
    for (const NamedDecl *declaration : unresolved->decls()) {
      const FunctionDecl *function =
          functionFromNamedDecl(declaration);
      if (function == nullptr)
        continue;
      sawCandidate = true;
      const PrimitiveKind candidate =
          classifyFunctionOrTemplate(function);
      if (candidate == PrimitiveKind::None)
        return PrimitiveKind::None;
      if (result != PrimitiveKind::None && result != candidate)
        return PrimitiveKind::None;
      result = candidate;
    }
    return sawCandidate ? result : PrimitiveKind::None;
  }

private:
  PrimitiveKind classifyFunctionOrTemplate(
      const FunctionDecl *function) {
    const auto templated =
        templateKinds.find(function->getCanonicalDecl());
    if (templated != templateKinds.end())
      return templated->second;
    return classify(function);
  }

  ASTContext &context;
  SourceManager &sourceManager;
  std::map<const FunctionDecl *, PrimitiveKind> functionKinds;
  std::map<const FunctionDecl *, PrimitiveKind> templateKinds;
  std::set<const FunctionDecl *> active;
};

bool exactlyTypeTemplateParameters(const FunctionDecl *function,
                                   unsigned count) {
  const FunctionTemplateDecl *functionTemplate =
      function->getDescribedFunctionTemplate();
  if (functionTemplate == nullptr)
    return false;
  const TemplateParameterList *parameters =
      functionTemplate->getTemplateParameters();
  if (parameters == nullptr || parameters->size() != count)
    return false;
  for (unsigned index = 0; index < count; ++index) {
    const auto *typeParameter =
        llvm::dyn_cast<TemplateTypeParmDecl>(
            parameters->getParam(index));
    if (typeParameter == nullptr ||
        typeParameter->isParameterPack() ||
        typeParameter->getIdentifier() == nullptr)
      return false;
  }
  return true;
}

bool adapterCallUsesParameter(const clang::CallExpr *call,
                              const ParmVarDecl *parameter) {
  if (call == nullptr || call->getNumArgs() != 3 ||
      !containsDecl(call->getCallee(), parameter))
    return false;
  const Expr *callee =
      stripExpr(call->getCallee());
  if (const auto *dependent =
          llvm::dyn_cast_or_null<
              clang::CXXDependentScopeMemberExpr>(
                  callee))
    return dependent->hasExplicitTemplateArgs() &&
           dependent->getNumTemplateArgs() == 1;
  const auto *member =
      llvm::dyn_cast_or_null<clang::MemberExpr>(
          callee);
  return member != nullptr &&
         member->hasExplicitTemplateArgs() &&
         member->getNumTemplateArgs() == 1;
}

bool expressionResultDiscarded(
    const Expr *expression,
    ASTContext &context) {
  const Stmt *current = expression;
  std::set<const Stmt *> active;
  while (current != nullptr &&
         active.insert(current).second) {
    const auto parents = context.getParents(*current);
    const Stmt *next = nullptr;
    for (const clang::DynTypedNode &parent : parents) {
      if (parent.get<clang::CompoundStmt>() != nullptr)
        return true;
      if (parent.get<clang::ReturnStmt>() != nullptr ||
          parent.get<VarDecl>() != nullptr)
        return false;
      if (const Stmt *statement = parent.get<Stmt>()) {
        if (!llvm::isa<Expr>(statement))
          return false;
        if (!llvm::isa<clang::ImplicitCastExpr,
                       clang::ParenExpr,
                       clang::ExprWithCleanups,
                       clang::CXXBindTemporaryExpr,
                       clang::MaterializeTemporaryExpr>(
                statement))
          return false;
        if (next == nullptr)
          next = statement;
      }
    }
    current = next;
  }
  return false;
}

class AdapterCallVisitor
    : public clang::RecursiveASTVisitor<AdapterCallVisitor> {
public:
  AdapterCallVisitor(const ParmVarDecl *load,
                     const ParmVarDecl *store,
                     ASTContext &context)
      : load(load), store(store), context(context) {}

  bool VisitCallExpr(clang::CallExpr *call) {
    if (discardedOrDeferredExecution(call, context))
      return true;
    sawLoad = sawLoad || adapterCallUsesParameter(call, load);
    sawStore = sawStore || adapterCallUsesParameter(call, store);
    return true;
  }

  bool proven() const { return sawLoad && sawStore; }

private:
  const ParmVarDecl *load;
  const ParmVarDecl *store;
  ASTContext &context;
  bool sawLoad = false;
  bool sawStore = false;
};

bool enumEquality(const Expr *condition,
                  const NonTypeTemplateParmDecl *&parameter,
                  const EnumConstantDecl *&constant) {
  parameter = nullptr;
  constant = nullptr;
  const auto *comparison =
      llvm::dyn_cast_or_null<BinaryOperator>(stripExpr(condition));
  if (comparison == nullptr ||
      comparison->getOpcode() != clang::BO_EQ)
    return false;
  parameter = directReferencedAs<NonTypeTemplateParmDecl>(
      comparison->getLHS());
  constant = directReferencedAs<EnumConstantDecl>(
      comparison->getRHS());
  if (parameter != nullptr && constant != nullptr)
    return parameter->getType()->isEnumeralType();
  parameter = directReferencedAs<NonTypeTemplateParmDecl>(
      comparison->getRHS());
  constant = directReferencedAs<EnumConstantDecl>(
      comparison->getLHS());
  return parameter != nullptr && constant != nullptr &&
         parameter->getType()->isEnumeralType();
}

bool simpleIntegralOffset(const Expr *expression) {
  expression = stripExpr(expression);
  if (expression == nullptr ||
      !expression->getType()->isIntegerType())
    return false;
  if (llvm::isa<clang::IntegerLiteral,
                clang::CharacterLiteral,
                DeclRefExpr>(expression))
    return true;
  if (const auto *unary =
          llvm::dyn_cast<clang::UnaryOperator>(expression)) {
    return (unary->getOpcode() == clang::UO_Plus ||
            unary->getOpcode() == clang::UO_Minus) &&
           simpleIntegralOffset(unary->getSubExpr());
  }
  const auto *binary =
      llvm::dyn_cast<BinaryOperator>(expression);
  if (binary == nullptr ||
      binary->isAssignmentOp() ||
      binary->getOpcode() == clang::BO_Comma)
    return false;
  switch (binary->getOpcode()) {
  case clang::BO_Mul:
  case clang::BO_Div:
  case clang::BO_Rem:
  case clang::BO_Add:
  case clang::BO_Sub:
  case clang::BO_Shl:
  case clang::BO_Shr:
  case clang::BO_And:
  case clang::BO_Xor:
  case clang::BO_Or:
    return simpleIntegralOffset(binary->getLHS()) &&
           simpleIntegralOffset(binary->getRHS());
  default:
    return false;
  }
}

const Decl *storageRootImpl(
    const Expr *expression,
    std::set<const VarDecl *> &active) {
  expression = stripExpr(expression);
  if (expression == nullptr)
    return nullptr;
  if (const auto *cast =
          llvm::dyn_cast<clang::ExplicitCastExpr>(
              expression)) {
    switch (cast->getCastKind()) {
    case clang::CK_NoOp:
    case clang::CK_BitCast:
    case clang::CK_AddressSpaceConversion:
      return storageRootImpl(
          cast->getSubExpr(), active);
    case clang::CK_Dependent:
      if (cast->getType()->isPointerType() &&
          (cast->getSubExpr()->getType()->isPointerType() ||
           cast->getSubExpr()->getType()->isArrayType()))
        return storageRootImpl(
            cast->getSubExpr(), active);
      return nullptr;
    default:
      return nullptr;
    }
  }
  if (const auto *subscript =
          llvm::dyn_cast<clang::ArraySubscriptExpr>(expression))
    return storageRootImpl(subscript->getBase(), active);
  if (const auto *member =
          llvm::dyn_cast<clang::MemberExpr>(expression))
    return storageRootImpl(member->getBase(), active);
  if (const auto *dependent =
          llvm::dyn_cast<clang::CXXDependentScopeMemberExpr>(
              expression))
    return storageRootImpl(dependent->getBase(), active);
  if (const auto *unary =
          llvm::dyn_cast<clang::UnaryOperator>(expression)) {
    if (unary->getOpcode() == clang::UO_AddrOf ||
        unary->getOpcode() == clang::UO_Deref)
      return storageRootImpl(unary->getSubExpr(), active);
  }
  if (const auto *binary =
          llvm::dyn_cast<BinaryOperator>(expression)) {
    if ((binary->getOpcode() == clang::BO_Add ||
         binary->getOpcode() == clang::BO_Sub)) {
      if (binary->getLHS()->getType()->isPointerType() &&
          !binary->getRHS()->getType()->isPointerType() &&
          simpleIntegralOffset(binary->getRHS()))
        return storageRootImpl(binary->getLHS(), active);
      if (binary->getOpcode() == clang::BO_Add &&
          binary->getRHS()->getType()->isPointerType() &&
          !binary->getLHS()->getType()->isPointerType() &&
          simpleIntegralOffset(binary->getLHS()))
        return storageRootImpl(binary->getRHS(), active);
      return nullptr;
    }
  }
  const auto *reference = llvm::dyn_cast<DeclRefExpr>(expression);
  if (reference == nullptr)
    return nullptr;
  const auto *variable =
      llvm::dyn_cast<VarDecl>(reference->getDecl());
  if (variable == nullptr)
    return reference->getDecl();
  if (!llvm::isa<ParmVarDecl>(variable) &&
      variable->getType()->isPointerType() &&
      variable->hasInit() && active.insert(variable).second) {
    const Decl *root =
        storageRootImpl(variable->getInit(), active);
    active.erase(variable);
    return root;
  }
  return variable;
}

const Decl *storageRoot(const Expr *expression) {
  std::set<const VarDecl *> active;
  return storageRootImpl(expression, active);
}

struct AdapterBufferEvidence {
  struct Access {
    const Decl *root = nullptr;
    const clang::CallExpr *call = nullptr;
  };
  std::set<const Decl *> loaded;
  std::set<const Decl *> stored;
  std::vector<Access> loadCalls;
  std::vector<Access> storeCalls;
};

class AdapterBufferVisitor
    : public clang::RecursiveASTVisitor<AdapterBufferVisitor> {
public:
  AdapterBufferVisitor(const ParmVarDecl *load,
                       const ParmVarDecl *store,
                       ASTContext &context)
      : load(load), store(store), context(context) {}

  bool VisitCallExpr(clang::CallExpr *call) {
    if (discardedOrDeferredExecution(call, context))
      return true;
    if (call->getNumArgs() == 0)
      return true;
    const Decl *root = storageRoot(call->getArg(0));
    if (root == nullptr)
      return true;
    if (adapterCallUsesParameter(call, load)) {
      evidence.loaded.insert(root);
      evidence.loadCalls.push_back({root, call});
    }
    if (adapterCallUsesParameter(call, store)) {
      evidence.stored.insert(root);
      evidence.storeCalls.push_back({root, call});
    }
    return true;
  }

  const ParmVarDecl *load;
  const ParmVarDecl *store;
  ASTContext &context;
  AdapterBufferEvidence evidence;
};

struct ProducedValue {
  const VarDecl *local = nullptr;
  const Expr *destination = nullptr;
  const BinaryOperator *assignment = nullptr;
  unsigned matches = 0;
};

const Stmt *producedStatement(
    const ProducedValue &produced) {
  if (produced.assignment != nullptr)
    return produced.assignment;
  return produced.local != nullptr &&
                 produced.local->hasInit()
             ? static_cast<const Stmt *>(
                   produced.local->getInit())
             : nullptr;
}

bool producedOnExecutablePath(
    const ProducedValue &produced,
    ASTContext &context) {
  const Stmt *statement =
      producedStatement(produced);
  return statement != nullptr &&
         !discardedOrDeferredExecution(
             statement, context);
}

class ProducerFinder
    : public clang::RecursiveASTVisitor<ProducerFinder> {
public:
  explicit ProducerFinder(const Expr *value)
      : value(stripExpr(value)) {}

  bool VisitVarDecl(VarDecl *variable) {
    if (variable->hasInit() &&
        stripExpr(variable->getInit()) == value) {
      ++produced.matches;
      produced.local = variable;
    }
    return true;
  }

  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (operation->getOpcode() == clang::BO_Assign &&
        stripExpr(operation->getRHS()) == value) {
      ++produced.matches;
      produced.destination = operation->getLHS();
      produced.assignment = operation;
    }
    return true;
  }

  const Expr *value;
  ProducedValue produced;
};

bool referencesProducedValue(const Expr *expression,
                             const ProducedValue &produced) {
  if (produced.local != nullptr)
    return directReferencedAs<VarDecl>(expression) ==
           produced.local;
  return produced.destination != nullptr &&
         sameReferenceExpression(
             expression, produced.destination);
}

class ExpResultUseVisitor
    : public clang::RecursiveASTVisitor<ExpResultUseVisitor> {
public:
  ExpResultUseVisitor(const ProducedValue &produced,
                      const Decl *bufferRoot,
                      ASTContext &context)
      : produced(produced), bufferRoot(bufferRoot),
        context(context) {}

  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (discardedOrDeferredExecution(
            operation, context))
      return true;
    if (operation->getOpcode() == clang::BO_Assign &&
        produced.local != nullptr &&
        referencesProducedValue(
            operation->getRHS(), produced) &&
        storageRoot(operation->getLHS()) == bufferRoot)
      ++bufferWrites;
    if (operation->getOpcode() == clang::BO_AddAssign &&
        referencesProducedValue(
            operation->getRHS(), produced)) {
      const Decl *root = storageRoot(operation->getLHS());
      if (root != nullptr) {
        ++accumulations;
        sumRoot = root;
        accumulationOperation = operation;
      }
    }
    return true;
  }

  const ProducedValue &produced;
  const Decl *bufferRoot;
  ASTContext &context;
  const Decl *sumRoot = nullptr;
  const BinaryOperator *accumulationOperation = nullptr;
  unsigned bufferWrites = 0;
  unsigned accumulations = 0;
};

struct SoftmaxExpEvidence {
  const Decl *bufferRoot = nullptr;
  const Decl *materializedRoot = nullptr;
  const Decl *shiftRoot = nullptr;
  const Decl *sumRoot = nullptr;
  const Stmt *operation = nullptr;
  const Stmt *accumulationOperation = nullptr;
};

struct SoftmaxNormalizeEvidence {
  const Decl *sourceRoot = nullptr;
  const Decl *inlineExpRoot = nullptr;
  const Decl *shiftRoot = nullptr;
  const Decl *outputRoot = nullptr;
  const Decl *sumRoot = nullptr;
  const BinaryOperator *operation = nullptr;
};

struct SoftmaxChoiceKey {
  const NonTypeTemplateParmDecl *parameter = nullptr;
  const EnumConstantDecl *constant = nullptr;

  bool operator<(const SoftmaxChoiceKey &other) const {
    return std::tie(parameter, constant) <
           std::tie(other.parameter, other.constant);
  }
};

using SoftmaxKernelBindings =
    std::map<const FunctionDecl *,
             std::set<SoftmaxChoiceKey>>;

class RootWriteCounter
    : public clang::RecursiveASTVisitor<RootWriteCounter> {
public:
  explicit RootWriteCounter(const Decl *root) : root(root) {}
  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (operation->isAssignmentOp() &&
        storageRoot(operation->getLHS()) == root)
      ++writes;
    return true;
  }
  const Decl *root;
  unsigned writes = 0;
};

class DirectExpAccumulationVisitor
    : public clang::RecursiveASTVisitor<
          DirectExpAccumulationVisitor> {
public:
  explicit DirectExpAccumulationVisitor(
      const clang::CallExpr *exponential,
      ASTContext &context)
      : exponential(exponential), context(context) {}

  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (discardedOrDeferredExecution(
            operation, context))
      return true;
    if (operation->getOpcode() == clang::BO_AddAssign &&
        stripExpr(operation->getRHS()) == exponential) {
      const Decl *root = storageRoot(operation->getLHS());
      if (root != nullptr) {
        ++matches;
        sumRoot = root;
        accumulationOperation = operation;
      }
    }
    return true;
  }

  const clang::CallExpr *exponential;
  ASTContext &context;
  const Decl *sumRoot = nullptr;
  const BinaryOperator *accumulationOperation = nullptr;
  unsigned matches = 0;
};

class PrimitiveCallCollector
    : public clang::RecursiveASTVisitor<PrimitiveCallCollector> {
public:
  PrimitiveCallCollector(
      PrimitiveRegistry &registry, ASTContext &context)
      : registry(registry), context(context) {}

  bool VisitCallExpr(clang::CallExpr *call) {
    if (discardedOrDeferredExecution(call, context))
      return true;
    // The injected facade handles only the exact built-in float compute
    // specialization.  Such arguments have no associated namespace, so ADL
    // contributes no candidates on the handled path.  Unsupported class
    // specializations fall through to the original CUDA implementation.
    const PrimitiveKind kind =
        registry.classifyCall(call, true);
    if (kind == PrimitiveKind::Exp)
      expCalls.push_back(call);
    else if (kind == PrimitiveKind::Divide)
      divideCalls.push_back(call);
    else if (kind == PrimitiveKind::Rsqrt)
      rsqrtCalls.push_back(call);
    return true;
  }

  PrimitiveRegistry &registry;
  ASTContext &context;
  std::vector<const clang::CallExpr *> expCalls;
  std::vector<const clang::CallExpr *> divideCalls;
  std::vector<const clang::CallExpr *> rsqrtCalls;
};

bool callHasDirectName(const clang::CallExpr *call,
                       llvm::StringRef name) {
  const FunctionDecl *callee =
      call == nullptr ? nullptr : call->getDirectCallee();
  return callee != nullptr && callee->getName() == name;
}

bool callCandidateFunctions(
    const clang::CallExpr *call,
    const FunctionDecl *caller,
    std::set<const FunctionDecl *> &result);

bool unresolvedFunctorCandidates(
    const clang::CallExpr *call,
    std::set<const FunctionDecl *> &functions);

const FunctionDecl *functionFromNamedDecl(
    const NamedDecl *declaration);

const TemplateTypeParmDecl *referencedTypeParameter(
    QualType type);

std::vector<clang::TemplateArgument>
callTemplateArguments(const clang::CallExpr *call,
                      const FunctionDecl *candidate);

const FunctionTemplateDecl *primaryFunctionTemplate(
    const FunctionDecl *function);

bool semanticCallCandidateFunctions(
    const clang::CallExpr *call,
    const FunctionDecl *caller,
    ASTContext &context,
    std::set<const FunctionDecl *> &candidates) {
  if (callCandidateFunctions(
          call, caller, candidates))
    return true;
  candidates.clear();
  if (unresolvedFunctorCandidates(
          call, candidates))
    return true;
  const auto *unresolved =
      llvm::dyn_cast_or_null<
          clang::UnresolvedLookupExpr>(
              stripExpr(
                  call == nullptr
                      ? nullptr
                      : call->getCallee()));
  if (unresolved == nullptr)
    return false;
  for (const NamedDecl *declaration :
       unresolved->decls()) {
    const FunctionDecl *function =
        functionFromNamedDecl(declaration);
    if (function == nullptr)
      return false;
    candidates.insert(
        function->getCanonicalDecl());
  }
  if (candidates.empty())
    return false;
  const SourceManager &sourceManager =
      context.getSourceManager();
  bool allSystem = true;
  for (const FunctionDecl *candidate : candidates) {
    allSystem =
        allSystem &&
        (sourceManager.isInSystemHeader(
             candidate->getLocation()) ||
         candidate->getBuiltinID() != 0);
  }
  if (allSystem)
    return true;
  return caller != nullptr &&
         caller->hasAttr<clang::CUDAGlobalAttr>() &&
         call->getNumArgs() == 1 &&
         templateTypeParameter(
             call->getArg(0)->getType(), 0, 2,
             false, false, false);
}

bool provenSideEffectFreeCall(
    const clang::CallExpr *call,
    const FunctionDecl *caller,
    ASTContext &context,
    std::set<const FunctionDecl *> &active);

class HelperEffectVisitor
    : public clang::RecursiveASTVisitor<
          HelperEffectVisitor> {
public:
  HelperEffectVisitor(
      const FunctionDecl *function,
      ASTContext &context,
      std::set<const FunctionDecl *> &active)
      : function(function), context(context),
        active(active) {}

  bool VisitVarDecl(VarDecl *variable) {
    if ((variable->isStaticLocal() &&
         !variable->hasAttr<
             clang::CUDASharedAttr>()) ||
        variable->getType().isVolatileQualified() ||
        variable->hasAttr<clang::CleanupAttr>())
      safe = false;
    QualType element =
        context.getBaseElementType(variable->getType());
    const CXXRecordDecl *record =
        element.isNull()
            ? nullptr
            : element->getAsCXXRecordDecl();
    if (record != nullptr &&
        !record->hasTrivialDestructor())
      safe = false;
    return safe;
  }

  bool VisitDeclRefExpr(DeclRefExpr *reference) {
    const auto *variable =
        llvm::dyn_cast<VarDecl>(
            reference->getDecl());
    if (variable != nullptr &&
        variable->getType().isVolatileQualified())
      safe = false;
    return safe;
  }

  bool VisitImplicitCastExpr(
      clang::ImplicitCastExpr *cast) {
    if (volatileLValueToRValue(cast))
      safe = false;
    return safe;
  }

  bool VisitAtomicExpr(clang::AtomicExpr *) {
    safe = false;
    return false;
  }

  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (!operation->isAssignmentOp())
      return true;
    const auto *variable =
        llvm::dyn_cast_or_null<VarDecl>(
            storageRoot(operation->getLHS()));
    const auto *parameter =
        llvm::dyn_cast_or_null<ParmVarDecl>(
            variable);
    if (variable == nullptr ||
        (parameter != nullptr &&
         (parameter->getType()->isPointerType() ||
          parameter->getType()->isReferenceType())) ||
        (variable->hasGlobalStorage() &&
         variable->getDeclContext() != function))
      safe = false;
    return safe;
  }

  bool VisitUnaryOperator(
      clang::UnaryOperator *operation) {
    switch (operation->getOpcode()) {
    case clang::UO_PreInc:
    case clang::UO_PostInc:
    case clang::UO_PreDec:
    case clang::UO_PostDec: {
      const auto *variable =
          llvm::dyn_cast_or_null<VarDecl>(
              storageRoot(operation->getSubExpr()));
      const auto *parameter =
          llvm::dyn_cast_or_null<ParmVarDecl>(
              variable);
      if (variable == nullptr ||
          (parameter != nullptr &&
           (parameter->getType()->isPointerType() ||
            parameter->getType()->isReferenceType())) ||
          (variable->hasGlobalStorage() &&
           variable->getDeclContext() != function))
        safe = false;
      break;
    }
    default:
      break;
    }
    return safe;
  }

  bool VisitCallExpr(clang::CallExpr *call) {
    if (!provenSideEffectFreeCall(
            call, function, context, active))
      safe = false;
    return safe;
  }

  bool VisitCXXConstructExpr(
      clang::CXXConstructExpr *construction) {
    const clang::CXXConstructorDecl *constructor =
        construction->getConstructor();
    if (constructor == nullptr || constructor->isTrivial())
      return safe;
    const CXXRecordDecl *record =
        constructor->getParent();
    const SourceManager &sourceManager =
        context.getSourceManager();
    if (record == nullptr ||
        record->getName() != "BlockReduce" ||
        !sourceManager.isInSystemHeader(
            record->getLocation())) {
      safe = false;
      return false;
    }
    for (const Expr *argument :
         construction->arguments()) {
      QualType type = argument->getType();
      if (!type->isPointerType() &&
          !type->isReferenceType())
        continue;
      const auto *variable =
          llvm::dyn_cast_or_null<VarDecl>(
              storageRoot(argument));
      if (variable == nullptr ||
          llvm::isa<ParmVarDecl>(variable) ||
          (variable->hasGlobalStorage() &&
           !variable->isStaticLocal())) {
        safe = false;
        return false;
      }
    }
    return safe;
  }

  bool VisitGCCAsmStmt(clang::GCCAsmStmt *) {
    safe = false;
    return false;
  }

  bool VisitMSAsmStmt(clang::MSAsmStmt *) {
    safe = false;
    return false;
  }

  bool VisitCXXThrowExpr(clang::CXXThrowExpr *) {
    safe = false;
    return false;
  }

  bool VisitCXXNewExpr(clang::CXXNewExpr *) {
    safe = false;
    return false;
  }

  bool VisitCXXDeleteExpr(clang::CXXDeleteExpr *) {
    safe = false;
    return false;
  }

  const FunctionDecl *function;
  ASTContext &context;
  std::set<const FunctionDecl *> &active;
  bool safe = true;
};

bool systemCallHasOnlyLocalPointerEffects(
    const clang::CallExpr *call,
    ASTContext &context) {
  for (const Expr *argument : call->arguments()) {
    QualType type =
        argument->getType().getNonReferenceType();
    if (type->isFunctionPointerType())
      continue;
    if (!type->isPointerType() &&
        !argument->getType()->isReferenceType())
      continue;
    const auto *variable =
        llvm::dyn_cast_or_null<VarDecl>(
            storageRoot(argument));
    if (variable == nullptr ||
        llvm::isa<ParmVarDecl>(variable) ||
        (variable->hasGlobalStorage() &&
         !variable->isStaticLocal()))
      return false;
  }
  return true;
}

bool exactPureSystemCall(
    const clang::CallExpr *call,
    const FunctionDecl *candidate,
    ASTContext &context) {
  if (call == nullptr || candidate == nullptr)
    return false;
  const SourceManager &sourceManager =
      context.getSourceManager();
  if (!sourceManager.isInSystemHeader(
          candidate->getLocation()) &&
      candidate->getBuiltinID() == 0)
    return false;
  if (!systemCallHasOnlyLocalPointerEffects(
          call, context))
    return false;
  const llvm::StringRef name = candidate->getName();
  if (name == "__syncthreads")
    return call->getNumArgs() == 0 &&
           call->getType()->isVoidType();
  static const char *const pureValueCalls[] = {
      "min",
      "max",
      "__shfl_sync",
      "__shfl_up_sync",
      "__shfl_down_sync",
      "__shfl_xor_sync",
  };
  for (const char *allowed : pureValueCalls) {
    if (name == allowed)
      return !call->getType()->isVoidType();
  }
  if (name != "Reduce" ||
      call->getType()->isVoidType())
    return false;
  const auto *method =
      llvm::dyn_cast<clang::CXXMethodDecl>(candidate);
  const CXXRecordDecl *record =
      method == nullptr ? nullptr : method->getParent();
  return record != nullptr &&
         record->getName() == "BlockReduce" &&
         sourceManager.isInSystemHeader(
             record->getLocation());
}

bool provenSideEffectFreeCall(
    const clang::CallExpr *call,
    const FunctionDecl *caller,
    ASTContext &context,
    std::set<const FunctionDecl *> &active) {
  std::set<const FunctionDecl *> candidates;
  if (!semanticCallCandidateFunctions(
          call, caller, context, candidates))
    return false;
  const SourceManager &sourceManager =
      context.getSourceManager();
  for (const FunctionDecl *candidate : candidates) {
    if (const FunctionDecl *definition =
            candidate->getDefinition())
      candidate = definition;
    const llvm::StringRef name = candidate->getName();
    if (name.contains("trap") ||
        name.starts_with("atomic"))
      return false;
    if (sourceManager.isInSystemHeader(
            candidate->getLocation()) ||
        candidate->getBuiltinID() != 0) {
      if (!exactPureSystemCall(
              call, candidate, context))
        return false;
      continue;
    }
    const FunctionDecl *canonical =
        candidate->getCanonicalDecl();
    if (!candidate->hasBody() ||
        !candidate->hasAttr<
            clang::CUDADeviceAttr>())
      return false;
    if (!active.insert(canonical).second)
      return false;
    HelperEffectVisitor effects(
        candidate, context, active);
    effects.TraverseStmt(
        const_cast<Stmt *>(candidate->getBody()));
    active.erase(canonical);
    if (!effects.safe)
      return false;
  }
  return true;
}

bool exactCudaExecutionBuiltinFetch(
    const clang::CallExpr *call,
    ASTContext &context);

bool positiveInfinityExpression(
    const Expr *expression,
    ASTContext &context) {
  Expr::EvalResult evaluated;
  if (expression != nullptr &&
      expression->EvaluateAsRValue(
          evaluated, context) &&
      evaluated.Val.isFloat())
    return evaluated.Val.getFloat().isInfinity() &&
           !evaluated.Val.getFloat().isNegative();
  const auto *call =
      llvm::dyn_cast_or_null<clang::CallExpr>(
          stripExpr(expression));
  if (call == nullptr || call->getNumArgs() != 1)
    return false;
  Expr::EvalResult bits;
  if (!call->getArg(0)->EvaluateAsInt(
          bits, context) ||
      !bits.Val.isInt())
    return false;
  const llvm::APSInt &integer = bits.Val.getInt();
  std::set<const FunctionDecl *> candidates;
  if (!semanticCallCandidateFunctions(
          call, nullptr, context, candidates) ||
      candidates.empty())
    return false;
  const SourceManager &sourceManager =
      context.getSourceManager();
  const bool floatPattern =
      integer.getLimitedValue() == 0x7f800000ULL &&
      context.hasSameType(
          call->getType().getCanonicalType(),
          context.FloatTy);
  const bool doublePattern =
      integer.getLimitedValue() ==
          0x7ff0000000000000ULL &&
      context.hasSameType(
          call->getType().getCanonicalType(),
          context.DoubleTy);
  if (!floatPattern && !doublePattern)
    return false;
  const llvm::StringRef expected =
      floatPattern ? "__int_as_float"
                   : "__longlong_as_double";
  for (const FunctionDecl *candidate : candidates) {
    if (candidate->getName() != expected ||
        (candidate->getBuiltinID() == 0 &&
         !sourceManager.isInSystemHeader(
             candidate->getLocation())))
      return false;
  }
  return true;
}

bool provenPositiveInfinityCall(
    const clang::CallExpr *call,
    const FunctionDecl *caller,
    ASTContext &context) {
  if (call == nullptr || call->getNumArgs() != 0)
    return false;
  const auto *unresolved =
      llvm::dyn_cast_or_null<
          clang::UnresolvedLookupExpr>(
              stripExpr(call->getCallee()));
  if (unresolved == nullptr ||
      !unresolved->hasExplicitTemplateArgs())
    return false;
  std::vector<clang::TemplateArgument> arguments;
  for (const clang::TemplateArgumentLoc &argument :
       unresolved->template_arguments())
    arguments.push_back(argument.getArgument());
  if (arguments.size() != 1 ||
      arguments[0].getKind() !=
          clang::TemplateArgument::Type ||
      !templateTypeParameter(
          arguments[0].getAsType(),
          0, 2, false, false, false))
    return false;

  const FunctionTemplateDecl *primary = nullptr;
  for (const NamedDecl *declaration :
       unresolved->decls()) {
    const FunctionDecl *function =
        functionFromNamedDecl(declaration);
    const FunctionTemplateDecl *candidate =
        llvm::dyn_cast<FunctionTemplateDecl>(
            declaration);
    if (candidate == nullptr &&
        function != nullptr)
      candidate = primaryFunctionTemplate(function);
    if (candidate == nullptr)
      return false;
    if (primary != nullptr &&
        primary->getCanonicalDecl() !=
            candidate->getCanonicalDecl())
      return false;
    primary = candidate;
  }
  const FunctionDecl *templated =
      primary == nullptr
          ? nullptr
          : primary->getTemplatedDecl();
  const TemplateParameterList *parameters =
      primary == nullptr
          ? nullptr
          : primary->getTemplateParameters();
  if (templated == nullptr || parameters == nullptr ||
      parameters->size() != 1 ||
      !llvm::isa<TemplateTypeParmDecl>(
          parameters->getParam(0)) ||
      templated->getNumParams() != 0 ||
      !templated->hasAttr<clang::CUDADeviceAttr>() ||
      !templateTypeParameter(
          templated->getReturnType(),
          0, 0, false, false, false))
    return false;

  bool hasFloat = false;
  bool hasDouble = false;
  unsigned specializations = 0;
  for (const FunctionDecl *specialization :
       primary->specializations()) {
    if (specialization->getTemplateSpecializationKind() !=
            clang::TSK_ExplicitSpecialization ||
        !specialization->hasBody() ||
        !specialization->hasAttr<
            clang::CUDADeviceAttr>() ||
        specialization->getNumParams() != 0)
      return false;
    const auto *returned =
        llvm::dyn_cast_or_null<clang::ReturnStmt>(
            stripAttributedStmt(
                onlyStatement(
                    specialization->getBody())));
    if (returned == nullptr ||
        !positiveInfinityExpression(
            returned->getRetValue(), context))
      return false;
    const QualType returnType =
        specialization->getReturnType()
            .getCanonicalType()
            .getUnqualifiedType();
    if (context.hasSameType(
            returnType,
            context.FloatTy))
      hasFloat = true;
    else if (context.hasSameType(
                 returnType,
                 context.DoubleTy))
      hasDouble = true;
    else
      return false;
    ++specializations;
  }
  (void)caller;
  return specializations == 2 &&
         hasFloat && hasDouble;
}

class SemanticEffectVisitor
    : public clang::RecursiveASTVisitor<
          SemanticEffectVisitor> {
public:
  SemanticEffectVisitor(
      const FunctionDecl *function,
      PrimitiveRegistry &registry,
      const std::set<const clang::CallExpr *>
          &allowedAdapterCalls,
      const std::set<const clang::CallExpr *>
          &allowedSemanticCalls,
      const ParmVarDecl *inverseOutput,
      const BinaryOperator *allowedInverseWrite,
      const SoftmaxChoiceKey *choice,
      ASTContext &context)
      : function(function), registry(registry),
        allowedAdapterCalls(allowedAdapterCalls),
        allowedSemanticCalls(allowedSemanticCalls),
        inverseOutput(inverseOutput),
        allowedInverseWrite(allowedInverseWrite),
        choice(choice),
        context(context) {}

  bool TraverseIfStmt(IfStmt *statement) {
    if (statement == nullptr)
      return true;
    TraverseStmt(statement->getInit());
    TraverseStmt(
        statement->getConditionVariableDeclStmt());
    TraverseStmt(statement->getCond());
    if (choice != nullptr) {
      const NonTypeTemplateParmDecl *parameter = nullptr;
      const EnumConstantDecl *constant = nullptr;
      if (enumEquality(
              statement->getCond(), parameter,
              constant) &&
          parameter == choice->parameter)
        return TraverseStmt(
            constant == choice->constant
                ? statement->getThen()
                : statement->getElse());
    }
    TraverseStmt(statement->getThen());
    TraverseStmt(statement->getElse());
    return safe;
  }

  bool VisitVarDecl(VarDecl *variable) {
    if ((variable->isStaticLocal() &&
         !variable->hasAttr<
             clang::CUDASharedAttr>()) ||
        variable->getType().isVolatileQualified() ||
        variable->hasAttr<clang::CleanupAttr>())
      safe = false;
    QualType element =
        context.getBaseElementType(variable->getType());
    const CXXRecordDecl *record =
        element.isNull()
            ? nullptr
            : element->getAsCXXRecordDecl();
    if (record != nullptr &&
        !record->hasTrivialDestructor())
      safe = false;
    return safe;
  }

  bool VisitDeclRefExpr(DeclRefExpr *reference) {
    const auto *variable =
        llvm::dyn_cast<VarDecl>(
            reference->getDecl());
    if (variable != nullptr &&
        variable->getType().isVolatileQualified())
      safe = false;
    return safe;
  }

  bool VisitImplicitCastExpr(
      clang::ImplicitCastExpr *cast) {
    if (volatileLValueToRValue(cast))
      safe = false;
    return safe;
  }

  bool VisitAtomicExpr(clang::AtomicExpr *) {
    safe = false;
    return false;
  }

  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (!operation->isAssignmentOp())
      return true;
    const auto *variable =
        llvm::dyn_cast_or_null<VarDecl>(
            storageRoot(operation->getLHS()));
    if (variable == nullptr ||
        ((variable->hasGlobalStorage() &&
          !variable->isConstexpr() &&
          !variable->hasAttr<
              clang::CUDASharedAttr>()) ||
         (llvm::isa<ParmVarDecl>(variable) &&
          (variable != inverseOutput ||
           operation != allowedInverseWrite))))
      safe = false;
    return safe;
  }

  bool VisitUnaryOperator(
      clang::UnaryOperator *operation) {
    switch (operation->getOpcode()) {
    case clang::UO_PreInc:
    case clang::UO_PostInc:
    case clang::UO_PreDec:
    case clang::UO_PostDec: {
      const auto *variable =
          llvm::dyn_cast_or_null<VarDecl>(
              storageRoot(operation->getSubExpr()));
      if (variable == nullptr ||
          ((variable->hasGlobalStorage() &&
           !variable->isConstexpr() &&
           !variable->hasAttr<
               clang::CUDASharedAttr>()) ||
           llvm::isa<ParmVarDecl>(variable)))
        safe = false;
      break;
    }
    default:
      break;
    }
    return safe;
  }

  bool VisitCallExpr(clang::CallExpr *call) {
    if (allowedAdapterCalls.count(call) != 0 ||
        allowedSemanticCalls.count(call) != 0 ||
        exactCudaExecutionBuiltinFetch(
            call, context) ||
        provenPositiveInfinityCall(
            call, function, context) ||
        registry.classifyCall(call, true) !=
            PrimitiveKind::None)
      return true;
    if (!provenSideEffectFreeCall(
            call, function, context, active))
      safe = false;
    return safe;
  }

  bool VisitCXXConstructExpr(
      clang::CXXConstructExpr *construction) {
    const clang::CXXConstructorDecl *constructor =
        construction->getConstructor();
    if (constructor != nullptr &&
        !constructor->isTrivial())
      safe = false;
    return safe;
  }

  bool VisitReturnStmt(clang::ReturnStmt *) {
    safe = false;
    return false;
  }

  bool VisitGotoStmt(clang::GotoStmt *) {
    safe = false;
    return false;
  }

  bool VisitIndirectGotoStmt(
      clang::IndirectGotoStmt *) {
    safe = false;
    return false;
  }

  bool VisitCXXTryStmt(clang::CXXTryStmt *) {
    safe = false;
    return false;
  }

  bool VisitGCCAsmStmt(clang::GCCAsmStmt *) {
    safe = false;
    return false;
  }

  bool VisitMSAsmStmt(clang::MSAsmStmt *) {
    safe = false;
    return false;
  }

  bool VisitCXXThrowExpr(clang::CXXThrowExpr *) {
    safe = false;
    return false;
  }

  bool VisitCXXNewExpr(clang::CXXNewExpr *) {
    safe = false;
    return false;
  }

  bool VisitCXXDeleteExpr(clang::CXXDeleteExpr *) {
    safe = false;
    return false;
  }

  const FunctionDecl *function;
  PrimitiveRegistry &registry;
  const std::set<const clang::CallExpr *>
      &allowedAdapterCalls;
  const std::set<const clang::CallExpr *>
      &allowedSemanticCalls;
  const ParmVarDecl *inverseOutput;
  const BinaryOperator *allowedInverseWrite;
  const SoftmaxChoiceKey *choice;
  ASTContext &context;
  std::set<const FunctionDecl *> active;
  bool safe = true;
};

bool semanticEffectsAllowed(
    const FunctionDecl *function,
    PrimitiveRegistry &registry,
    const std::set<const clang::CallExpr *>
        &allowedAdapterCalls,
    const std::set<const clang::CallExpr *>
        &allowedSemanticCalls,
    const ParmVarDecl *inverseOutput,
    const BinaryOperator *allowedInverseWrite,
    const SoftmaxChoiceKey *choice,
    ASTContext &context) {
  SemanticEffectVisitor effects(
      function, registry, allowedAdapterCalls,
      allowedSemanticCalls,
      inverseOutput,
      allowedInverseWrite, choice, context);
  effects.TraverseStmt(
      const_cast<Stmt *>(function->getBody()));
  return effects.safe;
}

class DivisionCollector
    : public clang::RecursiveASTVisitor<DivisionCollector> {
public:
  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (operation->getOpcode() == clang::BO_Div)
      divisions.push_back(operation);
    return true;
  }
  std::vector<const BinaryOperator *> divisions;
};

struct SoftmaxBranchEvidence {
  std::vector<SoftmaxExpEvidence> exponentials;
  std::vector<SoftmaxNormalizeEvidence> normalizations;
};

class ReferencedVariableSet
    : public clang::RecursiveASTVisitor<ReferencedVariableSet> {
public:
  bool VisitDeclRefExpr(DeclRefExpr *reference) {
    if (const auto *variable =
            llvm::dyn_cast<VarDecl>(reference->getDecl()))
      variables.insert(variable);
    return true;
  }
  std::set<const VarDecl *> variables;
};

bool expressionHasTemplateElementType(
    const Expr *expression,
    unsigned expectedDepth,
    unsigned expectedIndex) {
  expression = stripExpr(expression);
  if (expression == nullptr)
    return false;
  if (templateTypeParameter(
          expression->getType(), expectedDepth,
          expectedIndex, false, false, false))
    return true;
  const Expr *base = expression;
  if (const auto *subscript =
          llvm::dyn_cast<clang::ArraySubscriptExpr>(
              expression))
    base = stripExpr(subscript->getBase());
  if (templateElementType(
          base->getType(), expectedDepth,
          expectedIndex))
    return true;
  const auto *reference =
      llvm::dyn_cast<DeclRefExpr>(base);
  const auto *variable =
      reference == nullptr
          ? nullptr
          : llvm::dyn_cast<VarDecl>(
                reference->getDecl());
  return variable != nullptr && variable->hasInit() &&
         templateElementType(
             variable->getInit()->getType(),
             expectedDepth, expectedIndex);
}

bool cudaExecutionBuiltin(const VarDecl *variable) {
  if (variable == nullptr || variable->getIdentifier() == nullptr)
    return false;
  const llvm::StringRef name = variable->getName();
  if (name != "threadIdx" && name != "blockIdx" &&
      name != "blockDim" && name != "gridDim")
    return false;
  const CXXRecordDecl *record =
      variable->getType()
          .getNonReferenceType()
          .getUnqualifiedType()
          ->getAsCXXRecordDecl();
  return variable->hasGlobalStorage() &&
         !variable->isLocalVarDecl() &&
         record != nullptr &&
         record->getName().starts_with(
             "__cuda_builtin_");
}

bool variableDependsOnElementIndex(
    const VarDecl *variable,
    const std::set<const VarDecl *> &indices,
    std::set<const VarDecl *> &active) {
  if (variable == nullptr)
    return false;
  if (indices.count(variable) != 0 ||
      cudaExecutionBuiltin(variable))
    return true;
  if (!active.insert(variable).second)
    return false;
  bool depends = false;
  if (variable->hasInit()) {
    ReferencedVariableSet references;
    references.TraverseStmt(
        const_cast<Expr *>(variable->getInit()));
    for (const VarDecl *referenced :
         references.variables) {
      if (variableDependsOnElementIndex(
              referenced, indices, active)) {
        depends = true;
        break;
      }
    }
  }
  active.erase(variable);
  return depends;
}

const VarDecl *innermostForIndex(
    const Expr *expression,
    ASTContext &context) {
  const Stmt *current = expression;
  std::set<const Stmt *> active;
  while (current != nullptr &&
         active.insert(current).second) {
    const auto parents = context.getParents(*current);
    const Stmt *parentStatement = nullptr;
    for (const clang::DynTypedNode &parent : parents) {
      if (const auto *loop = parent.get<clang::ForStmt>()) {
        const auto *declaration =
            llvm::dyn_cast_or_null<clang::DeclStmt>(
                loop->getInit());
        if (declaration != nullptr &&
            declaration->isSingleDecl())
          return llvm::dyn_cast<VarDecl>(
              declaration->getSingleDecl());
      }
      if (parentStatement == nullptr)
        parentStatement = parent.get<Stmt>();
    }
    current = parentStatement;
  }
  return nullptr;
}

const FunctionDecl *enclosingFunction(
    const Stmt *statement,
    ASTContext &context) {
  const Stmt *current = statement;
  std::set<const Stmt *> active;
  while (current != nullptr &&
         active.insert(current).second) {
    const auto parents = context.getParents(*current);
    const Stmt *next = nullptr;
    for (const clang::DynTypedNode &parent : parents) {
      if (const auto *function =
              parent.get<FunctionDecl>())
        return function;
      if (next == nullptr)
        next = parent.get<Stmt>();
    }
    current = next;
  }
  return nullptr;
}

class AssignmentIndexDependencyVisitor
    : public clang::RecursiveASTVisitor<
          AssignmentIndexDependencyVisitor> {
public:
  AssignmentIndexDependencyVisitor(
      const VarDecl *wanted,
      const std::set<const VarDecl *> &indices,
      SourceLocation end,
      ASTContext &context)
      : wanted(wanted), indices(indices), end(end),
        context(context),
        sourceManager(context.getSourceManager()) {}

  bool before(SourceLocation location) const {
    return location.isValid() &&
           sourceManager.isBeforeInTranslationUnit(
               sourceManager.getExpansionLoc(location),
               sourceManager.getExpansionLoc(end));
  }

  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (!operation->isAssignmentOp() ||
        storageRoot(operation->getLHS()) != wanted ||
        !before(operation->getExprLoc()))
      return true;
    if (const VarDecl *loopIndex =
            innermostForIndex(operation, context)) {
      if (indices.count(loopIndex) != 0) {
        found = true;
        return false;
      }
    }
    ReferencedVariableSet references;
    references.TraverseStmt(
        const_cast<Expr *>(operation->getRHS()));
    for (const VarDecl *reference :
         references.variables) {
      std::set<const VarDecl *> active;
      if (variableDependsOnElementIndex(
              reference, indices, active)) {
        found = true;
        return false;
      }
    }
    return true;
  }

  bool VisitUnaryOperator(
      clang::UnaryOperator *operation) {
    if ((operation->getOpcode() ==
             clang::UO_PreInc ||
         operation->getOpcode() ==
             clang::UO_PostInc ||
         operation->getOpcode() ==
             clang::UO_PreDec ||
         operation->getOpcode() ==
             clang::UO_PostDec) &&
        storageRoot(operation->getSubExpr()) == wanted &&
        before(operation->getExprLoc()))
      found = true;
    return !found;
  }

  const VarDecl *wanted;
  const std::set<const VarDecl *> &indices;
  SourceLocation end;
  ASTContext &context;
  SourceManager &sourceManager;
  bool found = false;
};

const clang::ForStmt *enclosingForOf(
    const VarDecl *variable,
    ASTContext &context);

bool rowConstantShift(const Expr *value,
                      const Expr *shift,
                      ASTContext &context) {
  if (shift == nullptr || shift->HasSideEffects(context))
    return false;
  const auto *subscript =
      llvm::dyn_cast_or_null<clang::ArraySubscriptExpr>(
          stripExpr(value));
  ReferencedVariableSet indices;
  if (subscript != nullptr)
    indices.TraverseStmt(
        const_cast<Expr *>(subscript->getIdx()));
  if (const VarDecl *loopIndex =
          innermostForIndex(value, context))
    indices.variables.insert(loopIndex);
  if (indices.variables.empty())
    return false;
  ReferencedVariableSet shiftReferences;
  shiftReferences.TraverseStmt(
      const_cast<Expr *>(shift));
  const FunctionDecl *function =
      enclosingFunction(shift, context);
  for (const VarDecl *referenced :
       shiftReferences.variables) {
    std::set<const VarDecl *> active;
    if (variableDependsOnElementIndex(
            referenced, indices.variables, active))
      return false;
    if (!llvm::isa<ParmVarDecl>(referenced) &&
        function != nullptr &&
        (enclosingForOf(referenced, context) == nullptr ||
         indices.variables.count(referenced) != 0)) {
      AssignmentIndexDependencyVisitor assignments(
          referenced, indices.variables,
          shift->getBeginLoc(), context);
      assignments.TraverseStmt(
          const_cast<Stmt *>(function->getBody()));
      if (assignments.found)
        return false;
    }
  }
  return true;
}

SoftmaxBranchEvidence analyzeSoftmaxBranch(
    const Stmt *branch,
    PrimitiveRegistry &registry,
    ASTContext &context) {
  SoftmaxBranchEvidence result;
  PrimitiveCallCollector calls(registry, context);
  calls.TraverseStmt(const_cast<Stmt *>(branch));

  for (const clang::CallExpr *call : calls.expCalls) {
    if (call->getNumArgs() != 1)
      continue;
    const auto *shift =
        llvm::dyn_cast_or_null<BinaryOperator>(
            stripExpr(call->getArg(0)));
    if (shift == nullptr || shift->getOpcode() != clang::BO_Sub)
      continue;
    const Decl *bufferRoot = storageRoot(shift->getLHS());
    const Decl *shiftRoot = storageRoot(shift->getRHS());
    if (bufferRoot == nullptr ||
        shiftRoot == nullptr ||
        !rowConstantShift(
            shift->getLHS(), shift->getRHS(), context))
      continue;
    if (!expressionHasTemplateElementType(
            shift->getLHS(), 0, 2))
      continue;
    ProducerFinder producer(call);
    producer.TraverseStmt(const_cast<Stmt *>(branch));
    if (producer.produced.matches == 1 &&
        producedOnExecutablePath(
            producer.produced, context)) {
      if (producer.produced.destination != nullptr &&
          storageRoot(producer.produced.destination) !=
              bufferRoot)
        continue;
      ExpResultUseVisitor uses(
          producer.produced, bufferRoot, context);
      uses.TraverseStmt(const_cast<Stmt *>(branch));
      const unsigned expectedWrites =
          producer.produced.local == nullptr ? 0 : 1;
      if (uses.bufferWrites == expectedWrites &&
          uses.accumulations == 1 &&
          uses.sumRoot != nullptr) {
        result.exponentials.push_back(
            {bufferRoot, bufferRoot, shiftRoot,
             uses.sumRoot, call,
             uses.accumulationOperation});
      }
      continue;
    }
    if (producer.produced.matches != 0)
      continue;
    DirectExpAccumulationVisitor direct(
        call, context);
    direct.TraverseStmt(const_cast<Stmt *>(branch));
    if (direct.matches == 1 && direct.sumRoot != nullptr)
      result.exponentials.push_back(
          {bufferRoot, nullptr, shiftRoot,
           direct.sumRoot, call,
           direct.accumulationOperation});
  }

  auto recordDivision =
      [&](const Expr *value,
          const Expr *numerator,
          const Expr *denominator) {
        const Decl *sourceRoot = storageRoot(numerator);
        const Decl *inlineExpRoot = nullptr;
        const Decl *normalizationShiftRoot = nullptr;
        if (const auto *inlineExp =
                llvm::dyn_cast_or_null<clang::CallExpr>(
                    stripExpr(numerator))) {
          if (registry.classifyCall(inlineExp, true) !=
                  PrimitiveKind::Exp ||
              inlineExp->getNumArgs() != 1)
            return;
          const auto *shift =
              llvm::dyn_cast_or_null<BinaryOperator>(
                  stripExpr(inlineExp->getArg(0)));
          if (shift == nullptr ||
              shift->getOpcode() != clang::BO_Sub ||
              !rowConstantShift(
                  shift->getLHS(), shift->getRHS(),
                  context) ||
              !expressionHasTemplateElementType(
                  shift->getLHS(), 0, 2))
            return;
          inlineExpRoot = storageRoot(shift->getLHS());
          normalizationShiftRoot =
              storageRoot(shift->getRHS());
        }
        const Decl *sumRoot = storageRoot(denominator);
        if ((sourceRoot == nullptr &&
             inlineExpRoot == nullptr) ||
            sumRoot == nullptr)
          return;
        ProducerFinder producer(value);
        producer.TraverseStmt(const_cast<Stmt *>(branch));
        const Decl *outputRoot =
            storageRoot(producer.produced.destination);
        RootWriteCounter writes(outputRoot);
        writes.TraverseStmt(const_cast<Stmt *>(branch));
        if (producer.produced.matches == 1 &&
            producedOnExecutablePath(
                producer.produced, context) &&
            producer.produced.local == nullptr &&
            producer.produced.assignment != nullptr &&
            outputRoot != nullptr &&
            writes.writes == 1)
          result.normalizations.push_back(
              {sourceRoot, inlineExpRoot,
               normalizationShiftRoot,
               outputRoot, sumRoot,
               producer.produced.assignment});
      };

  for (const clang::CallExpr *call : calls.divideCalls) {
    if (call->getNumArgs() == 2)
      recordDivision(call, call->getArg(0), call->getArg(1));
  }
  DivisionCollector divisions;
  divisions.TraverseStmt(const_cast<Stmt *>(branch));
  for (const BinaryOperator *division : divisions.divisions) {
    if (discardedOrDeferredExecution(
            division, context))
      continue;
    recordDivision(
        division, division->getLHS(), division->getRHS());
  }
  return result;
}

class SoftmaxKernelVisitor
    : public clang::RecursiveASTVisitor<SoftmaxKernelVisitor> {
public:
  SoftmaxKernelVisitor(
      PrimitiveRegistry &registry, ASTContext &context)
      : registry(registry), context(context) {}

  bool VisitIfStmt(IfStmt *statement) {
    const NonTypeTemplateParmDecl *parameter = nullptr;
    const EnumConstantDecl *constant = nullptr;
    if (!enumEquality(
            statement->getCond(), parameter, constant))
      return true;
    const SoftmaxBranchEvidence branch =
        analyzeSoftmaxBranch(
            statement->getThen(), registry, context);
    SoftmaxBranchEvidence &combined =
        evidence[{parameter, constant}];
    combined.exponentials.insert(
        combined.exponentials.end(),
        branch.exponentials.begin(),
        branch.exponentials.end());
    combined.normalizations.insert(
        combined.normalizations.end(),
        branch.normalizations.begin(),
        branch.normalizations.end());
    return true;
  }

  PrimitiveRegistry &registry;
  ASTContext &context;
  std::map<SoftmaxChoiceKey, SoftmaxBranchEvidence> evidence;
};

class EnumUncontrolledExpCollector
    : public clang::RecursiveASTVisitor<
          EnumUncontrolledExpCollector> {
public:
  explicit EnumUncontrolledExpCollector(
      PrimitiveRegistry &registry)
      : registry(registry) {}

  bool VisitCallExpr(clang::CallExpr *call) {
    if (enumDepth == 0 &&
        registry.classifyCall(call, true) ==
            PrimitiveKind::Exp)
      calls.insert(call);
    return true;
  }

  bool TraverseIfStmt(IfStmt *statement) {
    if (statement == nullptr)
      return true;
    TraverseStmt(statement->getInit());
    TraverseStmt(statement->getConditionVariableDeclStmt());
    TraverseStmt(statement->getCond());
    const NonTypeTemplateParmDecl *parameter = nullptr;
    const EnumConstantDecl *constant = nullptr;
    const bool controlled = enumEquality(
        statement->getCond(), parameter, constant);
    if (controlled)
      ++enumDepth;
    TraverseStmt(statement->getThen());
    TraverseStmt(statement->getElse());
    if (controlled)
      --enumDepth;
    return true;
  }

  PrimitiveRegistry &registry;
  unsigned enumDepth = 0;
  std::set<const clang::CallExpr *> calls;
};

bool callHasStorageArgument(const clang::CallExpr *call,
                            const Decl *root) {
  if (call == nullptr || call->getNumArgs() != 1)
    return false;
  return storageRoot(call->getArg(0)) == root;
}

bool cudaThreadLaneZeroCondition(
    const Expr *condition,
    ASTContext &context);

bool reductionFunctorConstruction(
    const Expr *expression,
    const clang::TemplateTemplateParmDecl *parameter) {
  const auto *construction =
      llvm::dyn_cast_or_null<
          clang::CXXUnresolvedConstructExpr>(
              stripExpr(expression));
  const auto *specialization =
      construction == nullptr
          ? nullptr
          : construction->getType()
                .getNonReferenceType()
                .getUnqualifiedType()
                ->getAs<
                    clang::TemplateSpecializationType>();
  return construction != nullptr &&
         construction->getNumArgs() == 0 &&
         specialization != nullptr &&
         specialization->getTemplateName()
                 .getAsTemplateDecl() == parameter;
}

const clang::CXXMethodDecl *statelessReductionOperator(
    const clang::TemplateArgument &argument) {
  if (argument.getKind() !=
      clang::TemplateArgument::Template)
    return nullptr;
  const auto *classTemplate =
      llvm::dyn_cast_or_null<clang::ClassTemplateDecl>(
          argument.getAsTemplate().getAsTemplateDecl());
  const CXXRecordDecl *record =
      classTemplate == nullptr
          ? nullptr
          : classTemplate->getTemplatedDecl();
  if (record == nullptr ||
      !record->hasTrivialDefaultConstructor() ||
      !record->hasTrivialDestructor() ||
      !record->field_empty() ||
      record->getNumBases() != 0)
    return nullptr;
  const clang::CXXMethodDecl *operation = nullptr;
  for (const clang::CXXMethodDecl *method :
       record->methods()) {
    if (method->getOverloadedOperator() !=
        clang::OO_Call)
      continue;
    if (operation != nullptr)
      return nullptr;
    operation = method;
  }
  if (operation == nullptr ||
      !operation->hasBody() ||
      !operation->hasAttr<clang::CUDADeviceAttr>() ||
      operation->getNumParams() != 2)
    return nullptr;
  return operation;
}

bool provenAddFunctor(
    const clang::TemplateArgument &argument) {
  const clang::CXXMethodDecl *operation =
      statelessReductionOperator(argument);
  if (operation == nullptr)
    return false;
  const auto *returned =
      llvm::dyn_cast_or_null<clang::ReturnStmt>(
          stripAttributedStmt(
              onlyStatement(operation->getBody())));
  const auto *addition =
      llvm::dyn_cast_or_null<BinaryOperator>(
          stripExpr(
              returned == nullptr
                  ? nullptr
                  : returned->getRetValue()));
  if (addition == nullptr ||
      addition->getOpcode() != clang::BO_Add)
    return false;
  const ParmVarDecl *left =
      directReferencedAs<ParmVarDecl>(
          addition->getLHS());
  const ParmVarDecl *right =
      directReferencedAs<ParmVarDecl>(
          addition->getRHS());
  return (left == operation->getParamDecl(0) &&
          right == operation->getParamDecl(1)) ||
         (left == operation->getParamDecl(1) &&
          right == operation->getParamDecl(0));
}

bool exactSystemCallNamed(
    const clang::CallExpr *call,
    const FunctionDecl *caller,
    llvm::StringRef name,
    ASTContext &context) {
  std::set<const FunctionDecl *> candidates;
  if (!semanticCallCandidateFunctions(
          call, caller, context, candidates) ||
      candidates.empty())
    return false;
  for (const FunctionDecl *candidate : candidates) {
    if (candidate->getName() != name ||
        !exactPureSystemCall(
            call, candidate, context))
      return false;
  }
  return true;
}

bool provenMaximumFunctor(
    const clang::TemplateArgument &argument,
    ASTContext &context) {
  const clang::CXXMethodDecl *operation =
      statelessReductionOperator(argument);
  if (operation == nullptr)
    return false;
  const auto *returned =
      llvm::dyn_cast_or_null<clang::ReturnStmt>(
          stripAttributedStmt(
              onlyStatement(operation->getBody())));
  const auto *maximum =
      llvm::dyn_cast_or_null<clang::CallExpr>(
          stripExpr(
              returned == nullptr
                  ? nullptr
                  : returned->getRetValue()));
  if (maximum == nullptr ||
      maximum->getNumArgs() != 2 ||
      !exactSystemCallNamed(
          maximum, operation, "max", context))
    return false;
  const ParmVarDecl *left =
      directReferencedAs<ParmVarDecl>(
          maximum->getArg(0));
  const ParmVarDecl *right =
      directReferencedAs<ParmVarDecl>(
          maximum->getArg(1));
  return (left == operation->getParamDecl(0) &&
          right == operation->getParamDecl(1)) ||
         (left == operation->getParamDecl(1) &&
          right == operation->getParamDecl(0));
}

bool integerValue(const Expr *expression,
                  ASTContext &context,
                  uint64_t expected) {
  Expr::EvalResult result;
  return expression != nullptr &&
         expression->EvaluateAsInt(result, context) &&
         result.Val.isInt() &&
         result.Val.getInt().getLimitedValue() ==
             expected;
}

bool provenWarpReductionBody(
    const FunctionDecl *function,
    const clang::TemplateTemplateParmDecl *operation,
    const NonTypeTemplateParmDecl *width,
    ASTContext &context) {
  const auto *body =
      llvm::dyn_cast_or_null<clang::CompoundStmt>(
          function->getBody());
  if (body == nullptr || body->size() != 2 ||
      function->getNumParams() != 1)
    return false;
  auto statement = body->body_begin();
  const auto *loop =
      llvm::dyn_cast_or_null<clang::ForStmt>(
          stripAttributedStmt(*statement++));
  const auto *returned =
      llvm::dyn_cast_or_null<clang::ReturnStmt>(
          stripAttributedStmt(*statement));
  if (loop == nullptr || returned == nullptr ||
      directReferencedAs<ParmVarDecl>(
          returned->getRetValue()) !=
          function->getParamDecl(0))
    return false;
  const auto *declaration =
      llvm::dyn_cast_or_null<clang::DeclStmt>(
          loop->getInit());
  const auto *mask =
      declaration != nullptr &&
              declaration->isSingleDecl()
          ? llvm::dyn_cast<VarDecl>(
                declaration->getSingleDecl())
          : nullptr;
  const auto *initial =
      llvm::dyn_cast_or_null<BinaryOperator>(
          stripExpr(
              mask == nullptr
                  ? nullptr
                  : mask->getInit()));
  const auto *condition =
      llvm::dyn_cast_or_null<BinaryOperator>(
          stripExpr(loop->getCond()));
  const auto *increment =
      llvm::dyn_cast_or_null<
          clang::CompoundAssignOperator>(
              stripExpr(loop->getInc()));
  if (mask == nullptr ||
      !mask->getType()->isIntegerType() ||
      initial == nullptr ||
      initial->getOpcode() != clang::BO_Div ||
      directReferencedAs<
          NonTypeTemplateParmDecl>(
              initial->getLHS()) != width ||
      !integerValue(
          initial->getRHS(), context, 2) ||
      condition == nullptr ||
      condition->getOpcode() != clang::BO_GT ||
      directReferencedAs<VarDecl>(
          condition->getLHS()) != mask ||
      !evaluatesToZero(
          condition->getRHS(), context) ||
      increment == nullptr ||
      increment->getOpcode() != clang::BO_DivAssign ||
      directReferencedAs<VarDecl>(
          increment->getLHS()) != mask ||
      !integerValue(
          increment->getRHS(), context, 2))
    return false;
  const auto *assignment =
      llvm::dyn_cast_or_null<BinaryOperator>(
          stripAttributedStmt(
              onlyStatement(loop->getBody())));
  const auto *combine =
      llvm::dyn_cast_or_null<clang::CallExpr>(
          stripExpr(
              assignment == nullptr
                  ? nullptr
                  : assignment->getRHS()));
  if (assignment == nullptr ||
      assignment->getOpcode() != clang::BO_Assign ||
      directReferencedAs<ParmVarDecl>(
          assignment->getLHS()) !=
          function->getParamDecl(0) ||
      combine == nullptr ||
      combine->getNumArgs() != 2 ||
      !reductionFunctorConstruction(
          combine->getCallee(), operation) ||
      directReferencedAs<ParmVarDecl>(
          combine->getArg(0)) !=
          function->getParamDecl(0))
    return false;
  const auto *shuffle =
      llvm::dyn_cast_or_null<clang::CallExpr>(
          stripExpr(combine->getArg(1)));
  if (shuffle == nullptr ||
      (shuffle->getNumArgs() != 3 &&
       shuffle->getNumArgs() != 4) ||
      !exactSystemCallNamed(
          shuffle, function,
          "__shfl_xor_sync", context) ||
      !integerValue(
          shuffle->getArg(0), context,
          0xffffffffULL) ||
      directReferencedAs<ParmVarDecl>(
          shuffle->getArg(1)) !=
          function->getParamDecl(0) ||
      directReferencedAs<VarDecl>(
          shuffle->getArg(2)) != mask)
    return false;
  return shuffle->getNumArgs() == 3 ||
         directReferencedAs<
             NonTypeTemplateParmDecl>(
                 shuffle->getArg(3)) == width;
}

const Decl *singleDeclaration(
    const Stmt *statement) {
  const auto *declarations =
      llvm::dyn_cast_or_null<clang::DeclStmt>(
          stripAttributedStmt(statement));
  return declarations != nullptr &&
                 declarations->isSingleDecl()
             ? declarations->getSingleDecl()
             : nullptr;
}

bool provenBlockReductionBody(
    const FunctionDecl *function,
    const clang::TemplateTemplateParmDecl *operation,
    const TemplateTypeParmDecl *valueType,
    const NonTypeTemplateParmDecl *blockSize,
    ASTContext &context) {
  const auto *body =
      llvm::dyn_cast_or_null<clang::CompoundStmt>(
          function->getBody());
  if (body == nullptr || body->size() != 7 ||
      function->getNumParams() != 1)
    return false;
  auto statement = body->body_begin();
  const auto *alias =
      llvm::dyn_cast_or_null<clang::TypedefDecl>(
          singleDeclaration(*statement++));
  const auto *specialization =
      alias == nullptr
          ? nullptr
          : alias->getUnderlyingType()
                ->getAs<
                    clang::TemplateSpecializationType>();
  const auto *blockReduce =
      specialization == nullptr
          ? nullptr
          : llvm::dyn_cast_or_null<
                clang::ClassTemplateDecl>(
                specialization->getTemplateName()
                    .getAsTemplateDecl());
  const SourceManager &sourceManager =
      context.getSourceManager();
  llvm::ArrayRef<clang::TemplateArgument>
      specializationArguments;
  if (specialization != nullptr)
    specializationArguments =
        specialization->template_arguments();
  if (alias == nullptr || specialization == nullptr ||
      blockReduce == nullptr ||
      blockReduce->getName() != "BlockReduce" ||
      !sourceManager.isInSystemHeader(
          blockReduce->getLocation()) ||
      specializationArguments.size() != 2 ||
      specializationArguments[0].getKind() !=
          clang::TemplateArgument::Type ||
      referencedTypeParameter(
          specializationArguments[0].getAsType()) !=
          valueType ||
      specializationArguments[1].getKind() !=
          clang::TemplateArgument::Expression ||
      directReferencedAs<
          NonTypeTemplateParmDecl>(
              specializationArguments[1]
                  .getAsExpr()) != blockSize)
    return false;
  const auto *temporary =
      llvm::dyn_cast_or_null<VarDecl>(
          singleDeclaration(*statement++));
  const auto *broadcast =
      llvm::dyn_cast_or_null<VarDecl>(
          singleDeclaration(*statement++));
  const auto *result =
      llvm::dyn_cast_or_null<VarDecl>(
          singleDeclaration(*statement++));
  const auto *temporaryType =
      temporary == nullptr
          ? nullptr
          : temporary->getType()
                ->getAs<clang::DependentNameType>();
  const clang::Type *qualifier =
      temporaryType == nullptr
          ? nullptr
          : temporaryType->getQualifier()
                .getAsType();
  const auto *qualifierAlias =
      qualifier == nullptr
          ? nullptr
          : qualifier->getAs<clang::TypedefType>();
  if (temporary == nullptr ||
      broadcast == nullptr || result == nullptr ||
      !temporary->hasAttr<clang::CUDASharedAttr>() ||
      !broadcast->hasAttr<clang::CUDASharedAttr>() ||
      temporary->hasInit() || broadcast->hasInit() ||
      temporaryType == nullptr ||
      temporaryType->getIdentifier() == nullptr ||
      temporaryType->getIdentifier()->getName() !=
          "TempStorage" ||
      qualifierAlias == nullptr ||
      qualifierAlias->getDecl() != alias ||
      referencedTypeParameter(
          broadcast->getType()) != valueType ||
      referencedTypeParameter(
          result->getType()) != valueType ||
      !result->hasInit())
    return false;
  const auto *reduce =
      llvm::dyn_cast_or_null<clang::CallExpr>(
          stripExpr(result->getInit()));
  const auto *member =
      llvm::dyn_cast_or_null<
          clang::CXXDependentScopeMemberExpr>(
              stripExpr(
                  reduce == nullptr
                      ? nullptr
                      : reduce->getCallee()));
  const auto *construction =
      member == nullptr
          ? nullptr
          : llvm::dyn_cast_or_null<
                clang::CXXUnresolvedConstructExpr>(
                stripExpr(member->getBase()));
  const auto *constructionAlias =
      construction == nullptr
          ? nullptr
          : construction->getType()
                .getNonReferenceType()
                .getUnqualifiedType()
                ->getAs<clang::TypedefType>();
  if (reduce == nullptr ||
      reduce->getNumArgs() != 2 ||
      member == nullptr ||
      member->getMemberNameInfo().getAsString() !=
          "Reduce" ||
      construction == nullptr ||
      construction->getNumArgs() != 1 ||
      constructionAlias == nullptr ||
      constructionAlias->getDecl() != alias ||
      directReferencedAs<VarDecl>(
          construction->getArg(0)) != temporary ||
      directReferencedAs<ParmVarDecl>(
          reduce->getArg(0)) !=
          function->getParamDecl(0) ||
      !reductionFunctorConstruction(
          reduce->getArg(1), operation))
    return false;
  const auto *leader =
      llvm::dyn_cast_or_null<IfStmt>(
          stripAttributedStmt(*statement++));
  const auto *write =
      llvm::dyn_cast_or_null<BinaryOperator>(
          stripAttributedStmt(
              leader == nullptr
                  ? nullptr
                  : onlyStatement(
                        leader->getThen())));
  if (leader == nullptr ||
      leader->getElse() != nullptr ||
      !cudaThreadLaneZeroCondition(
          leader->getCond(), context) ||
      write == nullptr ||
      write->getOpcode() != clang::BO_Assign ||
      directReferencedAs<VarDecl>(
          write->getLHS()) != broadcast ||
      directReferencedAs<VarDecl>(
          write->getRHS()) != result)
    return false;
  const auto *sync =
      llvm::dyn_cast_or_null<clang::CallExpr>(
          stripExpr(
              llvm::dyn_cast_or_null<Expr>(
                  *statement++)));
  const auto *returned =
      llvm::dyn_cast_or_null<clang::ReturnStmt>(
          stripAttributedStmt(*statement));
  return sync != nullptr &&
         exactSystemCallNamed(
             sync, function,
             "__syncthreads", context) &&
         returned != nullptr &&
         directReferencedAs<VarDecl>(
             returned->getRetValue()) == broadcast;
}

enum class ProvenReductionKind {
  Sum,
  Maximum,
};

bool provenReductionCall(
    const clang::CallExpr *call,
    const FunctionDecl *caller,
    ProvenReductionKind kind,
    ASTContext &context) {
  if (call == nullptr || call->getNumArgs() != 1)
    return false;
  std::set<const FunctionDecl *> candidates;
  if (!semanticCallCandidateFunctions(
          call, caller, context, candidates) ||
      candidates.size() != 1)
    return false;
  const FunctionDecl *function =
      *candidates.begin();
  if (const FunctionDecl *definition =
          function->getDefinition())
    function = definition;
  const FunctionTemplateDecl *functionTemplate =
      primaryFunctionTemplate(function);
  const TemplateParameterList *parameters =
      functionTemplate == nullptr
          ? nullptr
          : functionTemplate->getTemplateParameters();
  const std::vector<clang::TemplateArgument> arguments =
      callTemplateArguments(call, function);
  if (parameters == nullptr ||
      parameters->size() != 3 ||
      arguments.size() != 3 ||
      !function->hasBody() ||
      !function->hasAttr<clang::CUDADeviceAttr>() ||
      referencedTypeParameter(
          function->getReturnType()) !=
          llvm::dyn_cast<TemplateTypeParmDecl>(
              parameters->getParam(1)) ||
      !templateTypeParameter(
          function->getParamDecl(0)->getType(),
          0, 1, false, false, false))
    return false;
  const bool functor =
      kind == ProvenReductionKind::Sum
          ? provenAddFunctor(arguments[0])
          : provenMaximumFunctor(
                arguments[0], context);
  if (!functor)
    return false;
  const auto *operation =
      llvm::dyn_cast<
          clang::TemplateTemplateParmDecl>(
              parameters->getParam(0));
  const auto *valueType =
      llvm::dyn_cast<TemplateTypeParmDecl>(
          parameters->getParam(1));
  const auto *width =
      llvm::dyn_cast<NonTypeTemplateParmDecl>(
          parameters->getParam(2));
  if (operation == nullptr || valueType == nullptr ||
      width == nullptr)
    return false;
  return provenWarpReductionBody(
             function, operation, width, context) ||
         provenBlockReductionBody(
             function, operation, valueType,
             width, context);
}

bool provenSumReductionCall(
    const clang::CallExpr *call,
    const FunctionDecl *caller,
    ASTContext &context) {
  return provenReductionCall(
      call, caller, ProvenReductionKind::Sum,
      context);
}

class StorageCallEdgeVisitor
    : public clang::RecursiveASTVisitor<StorageCallEdgeVisitor> {
public:
  StorageCallEdgeVisitor(
      const Decl *from, const Decl *to,
      const FunctionDecl *function,
      ASTContext &context)
      : from(from), to(to), function(function),
        context(context) {}

  bool VisitVarDecl(VarDecl *variable) {
    if (variable != to || !variable->hasInit())
      return true;
    const auto *call =
        llvm::dyn_cast_or_null<clang::CallExpr>(
            stripExpr(variable->getInit()));
    if (callHasStorageArgument(call, from) &&
        provenSumReductionCall(
            call, function, context) &&
        !discardedOrDeferredExecution(
            call, context))
      matches.push_back({call, nullptr});
    return true;
  }

  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (operation->getOpcode() != clang::BO_Assign ||
        storageRoot(operation->getLHS()) != to)
      return true;
    const auto *call =
        llvm::dyn_cast_or_null<clang::CallExpr>(
            stripExpr(operation->getRHS()));
    if (callHasStorageArgument(call, from) &&
        provenSumReductionCall(
            call, function, context) &&
        !discardedOrDeferredExecution(
            call, context))
      matches.push_back({call, operation});
    return true;
  }

  const Decl *from;
  const Decl *to;
  const FunctionDecl *function;
  ASTContext &context;
  struct Match {
    const clang::CallExpr *call = nullptr;
    const Stmt *producer = nullptr;
  };
  std::vector<Match> matches;
};

class StorageMutationRangeVisitor
    : public clang::RecursiveASTVisitor<
          StorageMutationRangeVisitor> {
public:
  StorageMutationRangeVisitor(
      const Decl *root,
      const Stmt *allowed,
      SourceLocation begin,
      SourceLocation end,
      SourceManager &sourceManager)
      : root(root), allowed(allowed), begin(begin),
        end(end), sourceManager(sourceManager) {}

  bool inside(SourceLocation location) const {
    location =
        sourceManager.getExpansionLoc(location);
    return sourceManager.isBeforeInTranslationUnit(
               begin, location) &&
           sourceManager.isBeforeInTranslationUnit(
               location, end);
  }

  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (operation != allowed &&
        operation->isAssignmentOp() &&
        storageRoot(operation->getLHS()) == root &&
        inside(operation->getExprLoc()))
      found = true;
    return !found;
  }

  bool VisitUnaryOperator(
      clang::UnaryOperator *operation) {
    switch (operation->getOpcode()) {
    case clang::UO_PreInc:
    case clang::UO_PostInc:
    case clang::UO_PreDec:
    case clang::UO_PostDec:
      if (storageRoot(operation->getSubExpr()) == root &&
          inside(operation->getExprLoc()))
        found = true;
      break;
    default:
      break;
    }
    return !found;
  }

  bool VisitCallExpr(clang::CallExpr *call) {
    if (!inside(call->getExprLoc()))
      return true;
    for (const Expr *argument : call->arguments()) {
      if (storageRoot(argument) == root &&
          (argument->getType()->isPointerType() ||
           argument->getType()->isReferenceType())) {
        found = true;
        break;
      }
    }
    return !found;
  }

  const Decl *root;
  const Stmt *allowed;
  SourceLocation begin;
  SourceLocation end;
  SourceManager &sourceManager;
  bool found = false;
};

class ReductionProducerVisitor
    : public clang::RecursiveASTVisitor<
          ReductionProducerVisitor> {
public:
  ReductionProducerVisitor(
      const Decl *destination,
      ProvenReductionKind kind,
      const FunctionDecl *function,
      ASTContext &context)
      : destination(destination), kind(kind),
        function(function), context(context) {}

  void record(const clang::CallExpr *call,
              const Stmt *producer) {
    if (call == nullptr || call->getNumArgs() != 1 ||
        storageRoot(call->getArg(0)) == nullptr ||
        !provenReductionCall(
            call, function, kind, context) ||
        discardedOrDeferredExecution(
            call, context))
      return;
    matches.push_back({call, producer});
  }

  bool VisitVarDecl(VarDecl *variable) {
    if (variable == destination &&
        variable->hasInit())
      record(
          llvm::dyn_cast_or_null<clang::CallExpr>(
              stripExpr(variable->getInit())),
          nullptr);
    return true;
  }

  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (operation->getOpcode() == clang::BO_Assign &&
        storageRoot(operation->getLHS()) ==
            destination)
      record(
          llvm::dyn_cast_or_null<clang::CallExpr>(
              stripExpr(operation->getRHS())),
          operation);
    return true;
  }

  struct Match {
    const clang::CallExpr *call = nullptr;
    const Stmt *producer = nullptr;
  };
  const Decl *destination;
  ProvenReductionKind kind;
  const FunctionDecl *function;
  ASTContext &context;
  std::vector<Match> matches;
};

bool storageProducedByReduction(
    const Stmt *body,
    const Decl *destination,
    const Stmt *use,
    ProvenReductionKind kind,
    const FunctionDecl *function,
    ASTContext &context,
    std::set<const clang::CallExpr *>
        *allowedCalls = nullptr) {
  if (body == nullptr || destination == nullptr ||
      use == nullptr)
    return false;
  ReductionProducerVisitor producer(
      destination, kind, function, context);
  producer.TraverseStmt(const_cast<Stmt *>(body));
  if (producer.matches.size() != 1)
    return false;
  const ReductionProducerVisitor::Match &match =
      producer.matches.front();
  SourceManager &sourceManager =
      context.getSourceManager();
  const SourceLocation begin =
      sourceManager.getExpansionLoc(
          match.call->getEndLoc());
  const SourceLocation end =
      sourceManager.getExpansionLoc(
          use->getBeginLoc());
  if (!sourceManager.isBeforeInTranslationUnit(
          begin, end))
    return false;
  StorageMutationRangeVisitor mutations(
      destination, match.producer, begin, end,
      sourceManager);
  mutations.TraverseStmt(const_cast<Stmt *>(body));
  if (mutations.found)
    return false;
  if (allowedCalls != nullptr)
    allowedCalls->insert(match.call);
  return true;
}

bool storageFlowsThroughCall(const Stmt *body,
                             const Decl *from,
                             const Decl *to,
                             const Stmt *use,
                             const FunctionDecl *function,
                             ASTContext &context,
                             std::set<const clang::CallExpr *>
                                 *allowedCalls = nullptr) {
  if (from == nullptr || to == nullptr ||
      use == nullptr)
    return false;
  StorageCallEdgeVisitor visitor(
      from, to, function, context);
  visitor.TraverseStmt(const_cast<Stmt *>(body));
  if (visitor.matches.size() != 1)
    return false;
  const StorageCallEdgeVisitor::Match &match =
      visitor.matches.front();
  SourceManager &sourceManager =
      context.getSourceManager();
  const SourceLocation begin =
      sourceManager.getExpansionLoc(
          match.call->getEndLoc());
  const SourceLocation end =
      sourceManager.getExpansionLoc(
          use->getBeginLoc());
  if (!sourceManager.isBeforeInTranslationUnit(
          begin, end))
    return false;
  StorageMutationRangeVisitor mutations(
      to, match.producer, begin, end,
      sourceManager);
  mutations.TraverseStmt(const_cast<Stmt *>(body));
  if (mutations.found)
    return false;
  if (allowedCalls != nullptr)
    allowedCalls->insert(match.call);
  return true;
}

bool hasAdapterAccessBefore(
    const std::vector<AdapterBufferEvidence::Access> &accesses,
    const Decl *root,
    const Stmt *operation,
    SourceManager &sourceManager) {
  if (operation == nullptr)
    return false;
  const SourceLocation operationLocation =
      sourceManager.getExpansionLoc(operation->getBeginLoc());
  for (const AdapterBufferEvidence::Access &access : accesses) {
    if (access.root == root && access.call != nullptr &&
        sourceManager.isBeforeInTranslationUnit(
            sourceManager.getExpansionLoc(
                access.call->getBeginLoc()),
            operationLocation))
      return true;
  }
  return false;
}

bool hasAdapterAccessBeforeLocation(
    const std::vector<AdapterBufferEvidence::Access> &accesses,
    const Decl *root,
    SourceLocation operationLocation,
    SourceManager &sourceManager) {
  operationLocation =
      sourceManager.getExpansionLoc(operationLocation);
  if (operationLocation.isInvalid())
    return false;
  for (const AdapterBufferEvidence::Access &access : accesses) {
    if (access.root == root && access.call != nullptr &&
        sourceManager.isBeforeInTranslationUnit(
            sourceManager.getExpansionLoc(
                access.call->getBeginLoc()),
            operationLocation))
      return true;
  }
  return false;
}

bool hasAdapterAccessAfter(
    const std::vector<AdapterBufferEvidence::Access> &accesses,
    const Decl *root,
    const Stmt *operation,
    SourceManager &sourceManager) {
  if (operation == nullptr)
    return false;
  const SourceLocation operationLocation =
      sourceManager.getExpansionLoc(operation->getEndLoc());
  for (const AdapterBufferEvidence::Access &access : accesses) {
    if (access.root == root && access.call != nullptr &&
        sourceManager.isBeforeInTranslationUnit(
            operationLocation,
            sourceManager.getExpansionLoc(
                access.call->getBeginLoc())))
      return true;
  }
  return false;
}

const Decl *directStorageReadRoot(const Expr *expression) {
  expression = stripExpr(expression);
  if (llvm::isa_and_nonnull<clang::ExplicitCastExpr>(
          expression))
    return nullptr;
  if (const auto *subscript =
          llvm::dyn_cast_or_null<clang::ArraySubscriptExpr>(
              expression))
    return storageRoot(subscript->getBase());
  if (const auto *member =
          llvm::dyn_cast_or_null<clang::MemberExpr>(expression))
    return storageRoot(member);
  if (const auto *dependent =
          llvm::dyn_cast_or_null<
              clang::CXXDependentScopeMemberExpr>(expression))
    return storageRoot(dependent);
  if (const auto *unary =
          llvm::dyn_cast_or_null<clang::UnaryOperator>(
              expression)) {
    if (unary->getOpcode() == clang::UO_Deref)
      return storageRoot(unary);
  }
  return llvm::isa_and_nonnull<DeclRefExpr>(expression)
             ? storageRoot(expression)
             : nullptr;
}

struct StorageCopyEdge {
  const Decl *from = nullptr;
  const Decl *to = nullptr;
  SourceLocation location;
};

class StorageCopyCollector
    : public clang::RecursiveASTVisitor<StorageCopyCollector> {
public:
  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (operation->getOpcode() != clang::BO_Assign)
      return true;
    const Decl *from =
        directStorageReadRoot(operation->getRHS());
    const Decl *to = storageRoot(operation->getLHS());
    if (from != nullptr && to != nullptr && from != to)
      edges.push_back({from, to, operation->getExprLoc()});
    return true;
  }

  std::vector<StorageCopyEdge> edges;
};

struct CanonicalForParts {
  const VarDecl *index = nullptr;
  const Expr *initial = nullptr;
  const Expr *bound = nullptr;
  const Expr *step = nullptr;
  clang::BinaryOperatorKind conditionOpcode =
      clang::BO_Comma;
  clang::BinaryOperatorKind incrementOpcode =
      clang::BO_Comma;
  bool unitIncrement = false;
};

const clang::ForStmt *enclosingForOf(
    const VarDecl *variable,
    ASTContext &context) {
  if (variable == nullptr)
    return nullptr;
  const Stmt *current = nullptr;
  for (const clang::DynTypedNode &parent :
       context.getParents(*variable)) {
    if (const auto *loop =
            parent.get<clang::ForStmt>())
      return loop;
    if (current == nullptr)
      current = parent.get<Stmt>();
  }
  std::set<const Stmt *> active;
  while (current != nullptr &&
         active.insert(current).second) {
    const auto parents = context.getParents(*current);
    const Stmt *next = nullptr;
    for (const clang::DynTypedNode &parent : parents) {
      if (const auto *loop =
              parent.get<clang::ForStmt>())
        return loop;
      if (next == nullptr)
        next = parent.get<Stmt>();
    }
    current = next;
  }
  return nullptr;
}

bool canonicalForParts(const clang::ForStmt *loop,
                       CanonicalForParts &parts) {
  if (loop == nullptr)
    return false;
  const auto *declaration =
      llvm::dyn_cast_or_null<clang::DeclStmt>(
          loop->getInit());
  if (declaration == nullptr ||
      !declaration->isSingleDecl())
    return false;
  const auto *index =
      llvm::dyn_cast<VarDecl>(
          declaration->getSingleDecl());
  const auto *condition =
      llvm::dyn_cast_or_null<BinaryOperator>(
          stripExpr(loop->getCond()));
  if (index == nullptr || !index->hasInit() ||
      condition == nullptr ||
      (condition->getOpcode() != clang::BO_LT &&
       condition->getOpcode() != clang::BO_LE) ||
      directReferencedAs<VarDecl>(
          condition->getLHS()) != index)
    return false;

  parts.index = index;
  parts.initial = index->getInit();
  parts.bound = condition->getRHS();
  parts.conditionOpcode = condition->getOpcode();
  const Expr *increment = stripExpr(loop->getInc());
  if (const auto *unary =
          llvm::dyn_cast_or_null<clang::UnaryOperator>(
              increment)) {
    if ((unary->getOpcode() == clang::UO_PreInc ||
         unary->getOpcode() == clang::UO_PostInc) &&
        directReferencedAs<VarDecl>(
            unary->getSubExpr()) == index) {
      parts.unitIncrement = true;
      return true;
    }
    return false;
  }
  const auto *compound =
      llvm::dyn_cast_or_null<clang::CompoundAssignOperator>(
          increment);
  if (compound == nullptr ||
      (compound->getOpcode() != clang::BO_AddAssign &&
       compound->getOpcode() != clang::BO_SubAssign) ||
      directReferencedAs<VarDecl>(
          compound->getLHS()) != index)
    return false;
  parts.incrementOpcode = compound->getOpcode();
  parts.step = compound->getRHS();
  return true;
}

bool canonicalForIndex(const VarDecl *variable,
                       ASTContext &context,
                       CanonicalForParts &parts) {
  return canonicalForParts(
             enclosingForOf(variable, context),
             parts) &&
         parts.index == variable;
}

struct AlphaExpressionContext {
  ASTContext &context;
  std::map<const VarDecl *, const VarDecl *> variables;
  std::set<std::pair<const VarDecl *, const VarDecl *>>
      activeLoops;
};

bool alphaEquivalentExpression(
    const Expr *lhs,
    const Expr *rhs,
    AlphaExpressionContext &alpha);

bool alphaEquivalentLoopIndices(
    const VarDecl *lhs,
    const VarDecl *rhs,
    AlphaExpressionContext &alpha) {
  const auto existing = alpha.variables.find(lhs);
  if (existing != alpha.variables.end())
    return existing->second == rhs;
  CanonicalForParts lhsParts;
  CanonicalForParts rhsParts;
  if (!canonicalForIndex(
          lhs, alpha.context, lhsParts) ||
      !canonicalForIndex(
          rhs, alpha.context, rhsParts))
    return false;
  const std::pair<const VarDecl *, const VarDecl *> pair{
      lhs, rhs};
  if (!alpha.activeLoops.insert(pair).second)
    return true;
  alpha.variables[lhs] = rhs;
  const bool shape =
      lhsParts.conditionOpcode ==
          rhsParts.conditionOpcode &&
      lhsParts.unitIncrement ==
          rhsParts.unitIncrement &&
      lhsParts.incrementOpcode ==
          rhsParts.incrementOpcode;
  const bool initial =
      alphaEquivalentExpression(
          lhsParts.initial, rhsParts.initial, alpha);
  const bool bound =
      alphaEquivalentExpression(
          lhsParts.bound, rhsParts.bound, alpha);
  const bool step =
      lhsParts.unitIncrement ||
      alphaEquivalentExpression(
          lhsParts.step, rhsParts.step, alpha);
  const bool same = shape && initial && bound && step;
  alpha.activeLoops.erase(pair);
  return same;
}

const clang::MemberExpr *cudaPseudoMember(
    const Expr *expression) {
  const auto *pseudo =
      llvm::dyn_cast_or_null<
          clang::PseudoObjectExpr>(
              stripExpr(expression));
  const auto *fetch =
      pseudo == nullptr
          ? nullptr
          : llvm::dyn_cast_or_null<
                clang::CallExpr>(
                stripExpr(
                    pseudo->getResultExpr()));
  return llvm::dyn_cast_or_null<
      clang::MemberExpr>(
      stripExpr(
          fetch == nullptr
              ? nullptr
              : fetch->getCallee()));
}

bool equivalentCudaPseudoObjects(
    const Expr *lhs,
    const Expr *rhs,
    ASTContext &context) {
  const clang::MemberExpr *lhsMember =
      cudaPseudoMember(lhs);
  const clang::MemberExpr *rhsMember =
      cudaPseudoMember(rhs);
  if (lhsMember == nullptr || rhsMember == nullptr ||
      lhsMember->getMemberDecl()->getCanonicalDecl() !=
          rhsMember->getMemberDecl()->getCanonicalDecl())
    return false;
  ReferencedVariableSet lhsReferences;
  ReferencedVariableSet rhsReferences;
  lhsReferences.TraverseStmt(
      const_cast<Expr *>(lhs));
  rhsReferences.TraverseStmt(
      const_cast<Expr *>(rhs));
  if (lhsReferences.variables.size() != 1 ||
      rhsReferences.variables.size() != 1)
    return false;
  const VarDecl *lhsBuiltin =
      *lhsReferences.variables.begin();
  const VarDecl *rhsBuiltin =
      *rhsReferences.variables.begin();
  const SourceManager &sourceManager =
      context.getSourceManager();
  return lhsBuiltin->getCanonicalDecl() ==
             rhsBuiltin->getCanonicalDecl() &&
         cudaExecutionBuiltin(lhsBuiltin) &&
         sourceManager.isInSystemHeader(
             lhsBuiltin->getLocation()) &&
         sourceManager.isInSystemHeader(
             lhsMember->getMemberDecl()->getLocation());
}

bool alphaEquivalentExpression(
    const Expr *lhs,
    const Expr *rhs,
    AlphaExpressionContext &alpha) {
  lhs = stripExpr(lhs);
  rhs = stripExpr(rhs);
  if (lhs == nullptr || rhs == nullptr)
    return lhs == rhs;
  if (sameReferenceExpression(lhs, rhs))
    return true;
  if (llvm::isa<clang::PseudoObjectExpr>(lhs) ||
      llvm::isa<clang::PseudoObjectExpr>(rhs))
    return equivalentCudaPseudoObjects(
        lhs, rhs, alpha.context);
  if (const auto *lhsCast =
          llvm::dyn_cast<clang::ExplicitCastExpr>(lhs)) {
    const auto *rhsCast =
        llvm::dyn_cast<clang::ExplicitCastExpr>(rhs);
    return rhsCast != nullptr &&
           lhsCast->getCastKind() ==
               rhsCast->getCastKind() &&
           alpha.context.hasSameType(
               lhsCast->getType().getCanonicalType(),
               rhsCast->getType().getCanonicalType()) &&
           alphaEquivalentExpression(
               lhsCast->getSubExpr(),
               rhsCast->getSubExpr(), alpha);
  }
  const auto *lhsReference =
      llvm::dyn_cast<DeclRefExpr>(lhs);
  const auto *rhsReference =
      llvm::dyn_cast<DeclRefExpr>(rhs);
  if (lhsReference != nullptr &&
      rhsReference != nullptr) {
    const auto *lhsVariable =
        llvm::dyn_cast<VarDecl>(
            lhsReference->getDecl());
    const auto *rhsVariable =
        llvm::dyn_cast<VarDecl>(
            rhsReference->getDecl());
    if (lhsVariable == nullptr || rhsVariable == nullptr)
      return lhsReference->getDecl() ==
             rhsReference->getDecl();
    if (alpha.variables.count(lhsVariable) != 0)
      return alpha.variables[lhsVariable] ==
             rhsVariable;
    CanonicalForParts lhsLoop;
    CanonicalForParts rhsLoop;
    const bool lhsIsLoop = canonicalForIndex(
        lhsVariable, alpha.context, lhsLoop);
    const bool rhsIsLoop = canonicalForIndex(
        rhsVariable, alpha.context, rhsLoop);
    if (lhsIsLoop || rhsIsLoop) {
      return alphaEquivalentLoopIndices(
          lhsVariable, rhsVariable, alpha);
    }
    if (lhsVariable->hasInit() &&
        rhsVariable->hasInit()) {
      return alphaEquivalentExpression(
          lhsVariable->getInit(),
          rhsVariable->getInit(), alpha);
    }
    return false;
  }
  const auto *lhsBinary =
      llvm::dyn_cast<BinaryOperator>(lhs);
  const auto *rhsBinary =
      llvm::dyn_cast<BinaryOperator>(rhs);
  if (lhsBinary != nullptr || rhsBinary != nullptr) {
    return lhsBinary != nullptr && rhsBinary != nullptr &&
           lhsBinary->getOpcode() ==
               rhsBinary->getOpcode() &&
           alphaEquivalentExpression(
               lhsBinary->getLHS(),
               rhsBinary->getLHS(), alpha) &&
           alphaEquivalentExpression(
               lhsBinary->getRHS(),
               rhsBinary->getRHS(), alpha);
  }
  const auto *lhsMember =
      llvm::dyn_cast<clang::MemberExpr>(lhs);
  const auto *rhsMember =
      llvm::dyn_cast<clang::MemberExpr>(rhs);
  if (lhsMember != nullptr || rhsMember != nullptr) {
    return lhsMember != nullptr && rhsMember != nullptr &&
           lhsMember->getMemberDecl()->getCanonicalDecl() ==
               rhsMember->getMemberDecl()->getCanonicalDecl() &&
           alphaEquivalentExpression(
               lhsMember->getBase(),
               rhsMember->getBase(), alpha);
  }
  const auto *lhsSubscript =
      llvm::dyn_cast<clang::ArraySubscriptExpr>(lhs);
  const auto *rhsSubscript =
      llvm::dyn_cast<clang::ArraySubscriptExpr>(rhs);
  if (lhsSubscript != nullptr || rhsSubscript != nullptr) {
    return lhsSubscript != nullptr &&
           rhsSubscript != nullptr &&
           alphaEquivalentExpression(
               lhsSubscript->getBase(),
               rhsSubscript->getBase(), alpha) &&
           alphaEquivalentExpression(
               lhsSubscript->getIdx(),
               rhsSubscript->getIdx(), alpha);
  }
  const auto *lhsUnary =
      llvm::dyn_cast<clang::UnaryOperator>(lhs);
  const auto *rhsUnary =
      llvm::dyn_cast<clang::UnaryOperator>(rhs);
  if (lhsUnary != nullptr || rhsUnary != nullptr) {
    if (lhsUnary == nullptr || rhsUnary == nullptr ||
        lhsUnary->getOpcode() != rhsUnary->getOpcode())
      return false;
    switch (lhsUnary->getOpcode()) {
    case clang::UO_Plus:
    case clang::UO_Minus:
    case clang::UO_Not:
    case clang::UO_LNot:
      return alphaEquivalentExpression(
          lhsUnary->getSubExpr(),
          rhsUnary->getSubExpr(), alpha);
    default:
      return false;
    }
  }
  const auto *lhsConditional =
      llvm::dyn_cast<clang::ConditionalOperator>(lhs);
  const auto *rhsConditional =
      llvm::dyn_cast<clang::ConditionalOperator>(rhs);
  if (lhsConditional != nullptr ||
      rhsConditional != nullptr) {
    return lhsConditional != nullptr &&
           rhsConditional != nullptr &&
           alphaEquivalentExpression(
               lhsConditional->getCond(),
               rhsConditional->getCond(), alpha) &&
           alphaEquivalentExpression(
               lhsConditional->getTrueExpr(),
               rhsConditional->getTrueExpr(), alpha) &&
           alphaEquivalentExpression(
               lhsConditional->getFalseExpr(),
               rhsConditional->getFalseExpr(), alpha);
  }
  const auto *lhsThis =
      llvm::dyn_cast<clang::CXXThisExpr>(lhs);
  const auto *rhsThis =
      llvm::dyn_cast<clang::CXXThisExpr>(rhs);
  if (lhsThis != nullptr || rhsThis != nullptr)
    return lhsThis != nullptr && rhsThis != nullptr &&
           alpha.context.hasSameType(
               lhsThis->getType().getCanonicalType(),
               rhsThis->getType().getCanonicalType());
  Expr::EvalResult lhsValue;
  Expr::EvalResult rhsValue;
  if (lhs->EvaluateAsInt(
          lhsValue, alpha.context) &&
      rhs->EvaluateAsInt(
          rhsValue, alpha.context) &&
      lhsValue.Val.isInt() && rhsValue.Val.isInt())
    return lhsValue.Val.getInt() ==
           rhsValue.Val.getInt();
  return false;
}

std::vector<clang::TemplateArgument>
memberTemplateArguments(const clang::CallExpr *call) {
  std::vector<clang::TemplateArgument> arguments;
  const Expr *callee =
      stripExpr(call == nullptr ? nullptr : call->getCallee());
  if (const auto *dependent =
          llvm::dyn_cast_or_null<
              clang::CXXDependentScopeMemberExpr>(callee)) {
    if (dependent->hasExplicitTemplateArgs()) {
      for (const clang::TemplateArgumentLoc &argument :
           dependent->template_arguments())
        arguments.push_back(argument.getArgument());
    }
  } else if (const auto *member =
                 llvm::dyn_cast_or_null<
                     clang::MemberExpr>(callee)) {
    if (member->hasExplicitTemplateArgs()) {
      for (const clang::TemplateArgumentLoc &argument :
           member->template_arguments())
        arguments.push_back(argument.getArgument());
    }
  }
  return arguments;
}

struct ControlBranch {
  const Expr *condition = nullptr;
  bool thenBranch = false;
};

std::vector<ControlBranch> enclosingControlBranches(
    const Stmt *statement,
    ASTContext &context) {
  std::vector<ControlBranch> branches;
  const Stmt *current = statement;
  std::set<const Stmt *> active;
  while (current != nullptr &&
         active.insert(current).second) {
    const auto parents = context.getParents(*current);
    const Stmt *next = nullptr;
    for (const clang::DynTypedNode &parent : parents) {
      if (const auto *conditional =
              parent.get<IfStmt>()) {
        if (conditional->getThen() == current)
          branches.push_back(
              {conditional->getCond(), true});
        else if (conditional->getElse() == current)
          branches.push_back(
              {conditional->getCond(), false});
      }
      if (next == nullptr)
        next = parent.get<Stmt>();
    }
    current = next;
  }
  return branches;
}

ControlBranch normalizedControlRequirement(
    ControlBranch branch,
    ASTContext &context) {
  branch.condition = stripExpr(branch.condition);
  bool changed = true;
  while (branch.condition != nullptr && changed) {
    changed = false;
    if (const auto *negation =
            llvm::dyn_cast<clang::UnaryOperator>(
                branch.condition)) {
      if (negation->getOpcode() == clang::UO_LNot) {
        branch.condition =
            stripExpr(negation->getSubExpr());
        branch.thenBranch = !branch.thenBranch;
        changed = true;
        continue;
      }
    }
    const auto *comparison =
        llvm::dyn_cast<BinaryOperator>(
            branch.condition);
    if (comparison == nullptr ||
        (comparison->getOpcode() != clang::BO_EQ &&
         comparison->getOpcode() != clang::BO_NE))
      continue;
    const Expr *atom = nullptr;
    if (evaluatesToZero(
            comparison->getLHS(), context))
      atom = comparison->getRHS();
    else if (evaluatesToZero(
                 comparison->getRHS(), context))
      atom = comparison->getLHS();
    if (atom == nullptr)
      continue;
    if (comparison->getOpcode() == clang::BO_EQ)
      branch.thenBranch = !branch.thenBranch;
    branch.condition = stripExpr(atom);
    changed = true;
  }
  return branch;
}

clang::BinaryOperatorKind invertedComparison(
    clang::BinaryOperatorKind opcode) {
  switch (opcode) {
  case clang::BO_EQ:
    return clang::BO_NE;
  case clang::BO_NE:
    return clang::BO_EQ;
  case clang::BO_LT:
    return clang::BO_GE;
  case clang::BO_LE:
    return clang::BO_GT;
  case clang::BO_GT:
    return clang::BO_LE;
  case clang::BO_GE:
    return clang::BO_LT;
  default:
    return clang::BO_Comma;
  }
}

struct EffectiveComparison {
  clang::BinaryOperatorKind opcode = clang::BO_Comma;
  const Expr *left = nullptr;
  const Expr *right = nullptr;
};

EffectiveComparison effectiveComparison(
    ControlBranch branch) {
  branch.condition = stripExpr(branch.condition);
  while (const auto *negation =
             llvm::dyn_cast_or_null<
                 clang::UnaryOperator>(
                     branch.condition)) {
    if (negation->getOpcode() != clang::UO_LNot)
      break;
    branch.condition =
        stripExpr(negation->getSubExpr());
    branch.thenBranch = !branch.thenBranch;
  }
  const auto *comparison =
      llvm::dyn_cast_or_null<BinaryOperator>(
          branch.condition);
  if (comparison == nullptr)
    return {};
  clang::BinaryOperatorKind opcode =
      comparison->getOpcode();
  if (invertedComparison(opcode) == clang::BO_Comma)
    return {};
  if (!branch.thenBranch)
    opcode = invertedComparison(opcode);
  return {
      opcode, comparison->getLHS(),
      comparison->getRHS()};
}

const Expr *canonicalAliasExpression(
    const Expr *expression,
    ASTContext &context,
    std::set<const VarDecl *> &active) {
  expression = stripExpr(expression);
  const auto *reference =
      llvm::dyn_cast_or_null<DeclRefExpr>(expression);
  const auto *variable =
      reference == nullptr
          ? nullptr
          : llvm::dyn_cast<VarDecl>(
                reference->getDecl());
  CanonicalForParts loopParts;
  if (variable == nullptr ||
      llvm::isa<ParmVarDecl>(variable) ||
      !variable->isLocalVarDecl() ||
      !variable->getType().isConstQualified() ||
      !variable->getType()->isIntegerType() ||
      !variable->hasInit() ||
      canonicalForIndex(
          variable, context, loopParts) ||
      !active.insert(variable).second)
    return expression;
  const Expr *result = canonicalAliasExpression(
      variable->getInit(), context, active);
  active.erase(variable);
  return result;
}

const Expr *canonicalAliasExpression(
    const Expr *expression,
    ASTContext &context) {
  std::set<const VarDecl *> active;
  return canonicalAliasExpression(
      expression, context, active);
}

bool canonicalDeclReference(
    const Expr *expression,
    const Decl *declaration,
    ASTContext &context) {
  return directReferencedDecl(
             canonicalAliasExpression(
                 expression, context)) ==
         declaration;
}

bool canonicalInteger(
    const Expr *expression,
    uint64_t expected,
    ASTContext &context) {
  return integerValue(
      canonicalAliasExpression(expression, context),
      context, expected);
}

bool canonicalCudaComponent(
    const Expr *expression,
    llvm::StringRef builtinName,
    llvm::StringRef component,
    ASTContext &context) {
  expression =
      canonicalAliasExpression(expression, context);
  const clang::MemberExpr *member =
      cudaPseudoMember(expression);
  if (member == nullptr)
    member =
        llvm::dyn_cast_or_null<clang::MemberExpr>(
            stripExpr(expression));
  if (member == nullptr)
    return false;
  const std::string memberName =
      member->getMemberNameInfo().getAsString();
  const std::string fetchName =
      std::string("__fetch_builtin_") +
      component.str();
  if (memberName != component &&
      memberName != fetchName)
    return false;
  ReferencedVariableSet references;
  references.TraverseStmt(
      const_cast<Expr *>(expression));
  if (references.variables.size() != 1)
    return false;
  const VarDecl *builtin =
      *references.variables.begin();
  const CXXRecordDecl *record =
      builtin->getType()
          .getNonReferenceType()
          .getUnqualifiedType()
          ->getAsCXXRecordDecl();
  const SourceManager &sourceManager =
      context.getSourceManager();
  return builtin->getName() == builtinName &&
         cudaExecutionBuiltin(builtin) &&
         record != nullptr &&
         sourceManager.isInSystemHeader(
             builtin->getLocation()) &&
         sourceManager.isInSystemHeader(
             record->getLocation()) &&
         sourceManager.isInSystemHeader(
             member->getMemberDecl()->getLocation());
}

bool canonicalBinary(
    const Expr *expression,
    clang::BinaryOperatorKind opcode,
    const Expr *&left,
    const Expr *&right,
    ASTContext &context) {
  const auto *binary =
      llvm::dyn_cast_or_null<BinaryOperator>(
          canonicalAliasExpression(
              expression, context));
  if (binary == nullptr ||
      binary->getOpcode() != opcode)
    return false;
  left = binary->getLHS();
  right = binary->getRHS();
  return true;
}

template<typename LeftMatcher, typename RightMatcher>
bool canonicalBinaryOperands(
    const Expr *expression,
    clang::BinaryOperatorKind opcode,
    LeftMatcher leftMatcher,
    RightMatcher rightMatcher,
    ASTContext &context,
    bool commutative = false) {
  const Expr *left = nullptr;
  const Expr *right = nullptr;
  if (!canonicalBinary(
          expression, opcode, left, right, context))
    return false;
  if (leftMatcher(left) && rightMatcher(right))
    return true;
  return commutative &&
         leftMatcher(right) && rightMatcher(left);
}

bool canonicalProductOfDecls(
    const Expr *expression,
    const Decl *left,
    const Decl *right,
    ASTContext &context) {
  return canonicalBinaryOperands(
      expression, clang::BO_Mul,
      [&](const Expr *operand) {
        return canonicalDeclReference(
            operand, left, context);
      },
      [&](const Expr *operand) {
        return canonicalDeclReference(
            operand, right, context);
      },
      context, true);
}

bool canonicalBuiltinProduct(
    const Expr *expression,
    llvm::StringRef leftBuiltin,
    llvm::StringRef leftComponent,
    llvm::StringRef rightBuiltin,
    llvm::StringRef rightComponent,
    ASTContext &context) {
  return canonicalBinaryOperands(
      expression, clang::BO_Mul,
      [&](const Expr *operand) {
        return canonicalCudaComponent(
            operand, leftBuiltin,
            leftComponent, context);
      },
      [&](const Expr *operand) {
        return canonicalCudaComponent(
            operand, rightBuiltin,
            rightComponent, context);
      },
      context, true);
}

bool statementNestedWithin(
    const Stmt *statement,
    const Stmt *ancestor,
    ASTContext &context) {
  if (statement == nullptr || ancestor == nullptr)
    return false;
  const Stmt *current = statement;
  std::set<const Stmt *> active;
  while (current != nullptr &&
         active.insert(current).second) {
    if (current == ancestor)
      return true;
    const auto parents = context.getParents(*current);
    const Stmt *next = nullptr;
    for (const clang::DynTypedNode &parent : parents) {
      if (next == nullptr)
        next = parent.get<Stmt>();
    }
    current = next;
  }
  return false;
}

enum class CanonicalKernelFamily {
  None,
  SoftmaxBlock,
  SoftmaxWarp,
  RmsBlock,
  RmsWarp,
};

struct CanonicalKernelShape {
  CanonicalKernelFamily family =
      CanonicalKernelFamily::None;
  const NonTypeTemplateParmDecl *pack = nullptr;
  const NonTypeTemplateParmDecl *blockSize = nullptr;
  const NonTypeTemplateParmDecl *columnsPerThread = nullptr;
  const NonTypeTemplateParmDecl *minimumColumnsPerThread =
      nullptr;
  const NonTypeTemplateParmDecl *threadGroupWidth = nullptr;
  const NonTypeTemplateParmDecl *rowsPerAccess = nullptr;
  const NonTypeTemplateParmDecl *padding = nullptr;
  const NonTypeTemplateParmDecl *algorithm = nullptr;
  const ParmVarDecl *rows = nullptr;
  const ParmVarDecl *columns = nullptr;
  const ParmVarDecl *inverseOutput = nullptr;

  bool softmax() const {
    return family == CanonicalKernelFamily::SoftmaxBlock ||
           family == CanonicalKernelFamily::SoftmaxWarp;
  }
  bool warp() const {
    return family == CanonicalKernelFamily::SoftmaxWarp ||
           family == CanonicalKernelFamily::RmsWarp;
  }
};

const NonTypeTemplateParmDecl *valueTemplateParameter(
    const TemplateParameterList *parameters,
    unsigned index) {
  return parameters != nullptr &&
                 index < parameters->size()
             ? llvm::dyn_cast<NonTypeTemplateParmDecl>(
                   parameters->getParam(index))
             : nullptr;
}

CanonicalKernelShape canonicalKernelShape(
    const FunctionDecl *function,
    bool softmax) {
  CanonicalKernelShape result;
  const FunctionTemplateDecl *functionTemplate =
      primaryFunctionTemplate(function);
  const TemplateParameterList *parameters =
      functionTemplate == nullptr
          ? nullptr
          : functionTemplate->getTemplateParameters();
  if (parameters == nullptr)
    return result;
  result.rows = function->getParamDecl(softmax ? 2 : 2);
  result.columns =
      function->getParamDecl(softmax ? 3 : 3);
  result.inverseOutput =
      softmax ? nullptr : function->getParamDecl(5);
  result.pack = valueTemplateParameter(parameters, 3);
  if (result.pack == nullptr)
    return {};
  if (softmax && parameters->size() == 6) {
    result.family =
        CanonicalKernelFamily::SoftmaxBlock;
    result.blockSize =
        valueTemplateParameter(parameters, 4);
    result.algorithm =
        valueTemplateParameter(parameters, 5);
  } else if (softmax && parameters->size() == 9) {
    result.family =
        CanonicalKernelFamily::SoftmaxWarp;
    result.columnsPerThread =
        valueTemplateParameter(parameters, 4);
    result.threadGroupWidth =
        valueTemplateParameter(parameters, 5);
    result.rowsPerAccess =
        valueTemplateParameter(parameters, 6);
    result.padding =
        valueTemplateParameter(parameters, 7);
    result.algorithm =
        valueTemplateParameter(parameters, 8);
  } else if (!softmax && parameters->size() == 5) {
    result.family =
        CanonicalKernelFamily::RmsBlock;
    result.blockSize =
        valueTemplateParameter(parameters, 4);
  } else if (!softmax && parameters->size() == 9) {
    result.family =
        CanonicalKernelFamily::RmsWarp;
    result.columnsPerThread =
        valueTemplateParameter(parameters, 4);
    result.minimumColumnsPerThread =
        valueTemplateParameter(parameters, 5);
    result.threadGroupWidth =
        valueTemplateParameter(parameters, 6);
    result.rowsPerAccess =
        valueTemplateParameter(parameters, 7);
    result.padding =
        valueTemplateParameter(parameters, 8);
  }
  if (result.family == CanonicalKernelFamily::None ||
      (result.warp() &&
       (result.columnsPerThread == nullptr ||
        result.threadGroupWidth == nullptr ||
        result.rowsPerAccess == nullptr ||
        result.padding == nullptr)) ||
      (!result.warp() &&
       result.blockSize == nullptr) ||
      (result.softmax() &&
       result.algorithm == nullptr) ||
      (result.family == CanonicalKernelFamily::RmsWarp &&
       result.minimumColumnsPerThread == nullptr))
    return {};
  return result;
}

bool canonicalBlockRowLoop(
    const clang::ForStmt *loop,
    const CanonicalKernelShape &shape,
    ASTContext &context) {
  CanonicalForParts parts;
  return canonicalForParts(loop, parts) &&
         parts.conditionOpcode == clang::BO_LT &&
         !parts.unitIncrement &&
         parts.incrementOpcode ==
             clang::BO_AddAssign &&
         canonicalCudaComponent(
             parts.initial, "blockIdx", "x",
             context) &&
         canonicalDeclReference(
             parts.bound, shape.rows, context) &&
         canonicalCudaComponent(
             parts.step, "gridDim", "x",
             context);
}

bool canonicalBlockPackLoop(
    const clang::ForStmt *loop,
    const CanonicalKernelShape &shape,
    ASTContext &context) {
  CanonicalForParts parts;
  if (!canonicalForParts(loop, parts) ||
      parts.conditionOpcode != clang::BO_LT ||
      parts.unitIncrement ||
      parts.incrementOpcode !=
          clang::BO_AddAssign ||
      !canonicalCudaComponent(
          parts.initial, "threadIdx", "x",
          context) ||
      !canonicalDeclReference(
          parts.step, shape.blockSize, context))
    return false;
  return canonicalBinaryOperands(
      parts.bound, clang::BO_Div,
      [&](const Expr *operand) {
        return canonicalDeclReference(
            operand, shape.columns, context);
      },
      [&](const Expr *operand) {
        return canonicalDeclReference(
            operand, shape.pack, context);
      },
      context);
}

const VarDecl *canonicalPackedColumnIndex(
    const Expr *expression,
    const NonTypeTemplateParmDecl *pack,
    ASTContext &context) {
  const Expr *left = nullptr;
  const Expr *right = nullptr;
  if (!canonicalBinary(
          expression, clang::BO_Mul,
          left, right, context))
    return nullptr;
  const Expr *indexExpression = nullptr;
  if (canonicalDeclReference(right, pack, context))
    indexExpression = left;
  else if (canonicalDeclReference(left, pack, context))
    indexExpression = right;
  const auto *indexReference =
      llvm::dyn_cast_or_null<DeclRefExpr>(
          canonicalAliasExpression(
              indexExpression, context));
  return indexReference == nullptr
             ? nullptr
             : llvm::dyn_cast<VarDecl>(
                   indexReference->getDecl());
}

bool canonicalWarpGroupExpression(
    const Expr *expression,
    bool grid,
    ASTContext &context) {
  if (grid) {
    return canonicalBuiltinProduct(
        expression, "gridDim", "x",
        "blockDim", "y", context);
  }
  const Expr *left = nullptr;
  const Expr *right = nullptr;
  if (!canonicalBinary(
          expression, clang::BO_Add,
          left, right, context))
    return false;
  return canonicalBuiltinProduct(
             left, "blockIdx", "x",
             "blockDim", "y", context) &&
         canonicalCudaComponent(
             right, "threadIdx", "y",
             context);
}

bool canonicalCeilRowGroups(
    const Expr *expression,
    const CanonicalKernelShape &shape,
    ASTContext &context) {
  const Expr *numerator = nullptr;
  const Expr *denominator = nullptr;
  if (!canonicalBinary(
          expression, clang::BO_Div,
          numerator, denominator, context) ||
      !canonicalDeclReference(
          denominator, shape.rowsPerAccess,
          context))
    return false;
  const Expr *sum = nullptr;
  const Expr *one = nullptr;
  if (!canonicalBinary(
          numerator, clang::BO_Sub,
          sum, one, context) ||
      !canonicalInteger(one, 1, context))
    return false;
  return canonicalBinaryOperands(
      sum, clang::BO_Add,
      [&](const Expr *operand) {
        return canonicalDeclReference(
            operand, shape.rows, context);
      },
      [&](const Expr *operand) {
        return canonicalDeclReference(
            operand, shape.rowsPerAccess,
            context);
      },
      context, true);
}

bool canonicalWarpOuterLoop(
    const clang::ForStmt *loop,
    const CanonicalKernelShape &shape,
    ASTContext &context) {
  CanonicalForParts parts;
  if (!canonicalForParts(loop, parts) ||
      parts.conditionOpcode != clang::BO_LT ||
      parts.unitIncrement ||
      parts.incrementOpcode !=
          clang::BO_AddAssign)
    return false;
  if (shape.family ==
      CanonicalKernelFamily::SoftmaxWarp) {
    const bool initial =
        canonicalBinaryOperands(
            parts.initial, clang::BO_Mul,
            [&](const Expr *operand) {
              return canonicalWarpGroupExpression(
                  operand, false, context);
            },
            [&](const Expr *operand) {
              return canonicalDeclReference(
                  operand, shape.rowsPerAccess,
                  context);
            },
            context, true);
    const bool step =
        canonicalBinaryOperands(
            parts.step, clang::BO_Mul,
            [&](const Expr *operand) {
              return canonicalWarpGroupExpression(
                  operand, true, context);
            },
            [&](const Expr *operand) {
              return canonicalDeclReference(
                  operand, shape.rowsPerAccess,
                  context);
            },
            context, true);
    return initial && step &&
           canonicalDeclReference(
               parts.bound, shape.rows, context);
  }
  return canonicalWarpGroupExpression(
             parts.initial, false, context) &&
         canonicalWarpGroupExpression(
             parts.step, true, context) &&
         canonicalCeilRowGroups(
             parts.bound, shape, context);
}

bool canonicalUnitLoop(
    const clang::ForStmt *loop,
    const std::function<bool(const Expr *)>
        &initial,
    const std::function<bool(const Expr *)>
        &bound) {
  CanonicalForParts parts;
  return canonicalForParts(loop, parts) &&
         parts.conditionOpcode == clang::BO_LT &&
         parts.unitIncrement &&
         initial(parts.initial) &&
         bound(parts.bound);
}

bool canonicalWarpColumn(
    const Expr *expression,
    const VarDecl *packIndex,
    const CanonicalKernelShape &shape,
    ASTContext &context) {
  const Expr *laneGroup = nullptr;
  const Expr *pack = nullptr;
  if (!canonicalBinary(
          expression, clang::BO_Mul,
          laneGroup, pack, context) ||
      !canonicalDeclReference(
          pack, shape.pack, context))
    return false;
  const Expr *packGroup = nullptr;
  const Expr *lane = nullptr;
  if (!canonicalBinary(
          laneGroup, clang::BO_Add,
          packGroup, lane, context))
    return false;
  return canonicalProductOfDecls(
             packGroup, packIndex,
             shape.threadGroupWidth, context) &&
         canonicalCudaComponent(
             lane, "threadIdx", "x", context);
}

bool canonicalSoftmaxWarpRow(
    const Expr *expression,
    const VarDecl *outerRow,
    const VarDecl *rowInGroup,
    ASTContext &context) {
  return canonicalBinaryOperands(
      expression, clang::BO_Add,
      [&](const Expr *operand) {
        return canonicalDeclReference(
            operand, outerRow, context);
      },
      [&](const Expr *operand) {
        return canonicalDeclReference(
            operand, rowInGroup, context);
      },
      context, true);
}

bool canonicalRmsWarpRow(
    const Expr *expression,
    const VarDecl *outerRow,
    const VarDecl *rowInGroup,
    const CanonicalKernelShape &shape,
    ASTContext &context) {
  const Expr *group = nullptr;
  const Expr *inner = nullptr;
  if (!canonicalBinary(
          expression, clang::BO_Add,
          group, inner, context))
    return false;
  return canonicalProductOfDecls(
             group, outerRow,
             shape.rowsPerAccess, context) &&
         canonicalDeclReference(
             inner, rowInGroup, context);
}

bool canonicalAdapterPackArgument(
    const clang::CallExpr *call,
    const CanonicalKernelShape &shape,
    ASTContext &context) {
  const std::vector<clang::TemplateArgument> arguments =
      memberTemplateArguments(call);
  return arguments.size() == 1 &&
         arguments[0].getKind() ==
             clang::TemplateArgument::Expression &&
         canonicalDeclReference(
             arguments[0].getAsExpr(),
             shape.pack, context);
}

bool canonicalBlockAdapterAccesses(
    const AdapterBufferEvidence &adapters,
    const CanonicalKernelShape &shape,
    ASTContext &context) {
  const clang::ForStmt *rowLoop = nullptr;
  auto prove =
      [&](const AdapterBufferEvidence::Access &access) {
        const clang::CallExpr *call = access.call;
        if (call == nullptr || call->getNumArgs() != 3 ||
            !canonicalAdapterPackArgument(
                call, shape, context))
          return false;
        const auto *rowReference =
            llvm::dyn_cast_or_null<DeclRefExpr>(
                canonicalAliasExpression(
                    call->getArg(1), context));
        const auto *row =
            rowReference == nullptr
                ? nullptr
                : llvm::dyn_cast<VarDecl>(
                      rowReference->getDecl());
        const clang::ForStmt *candidateRowLoop =
            enclosingForOf(row, context);
        if (row == nullptr ||
            !canonicalBlockRowLoop(
                candidateRowLoop, shape, context) ||
            !statementNestedWithin(
                call, candidateRowLoop, context) ||
            (rowLoop != nullptr &&
             rowLoop != candidateRowLoop))
          return false;
        rowLoop = candidateRowLoop;
        const VarDecl *packIndex =
            canonicalPackedColumnIndex(
                call->getArg(2), shape.pack,
                context);
        const clang::ForStmt *packLoop =
            enclosingForOf(packIndex, context);
        return packIndex != nullptr &&
               canonicalBlockPackLoop(
                   packLoop, shape, context) &&
               statementNestedWithin(
                   call, packLoop, context) &&
               statementNestedWithin(
                   packLoop, candidateRowLoop,
                   context);
      };
  for (const AdapterBufferEvidence::Access &access :
       adapters.loadCalls) {
    if (!prove(access))
      return false;
  }
  for (const AdapterBufferEvidence::Access &access :
       adapters.storeCalls) {
    if (!prove(access))
      return false;
  }
  return rowLoop != nullptr;
}

bool canonicalWarpAdapterAccesses(
    const AdapterBufferEvidence &adapters,
    const CanonicalKernelShape &shape,
    ASTContext &context) {
  const clang::ForStmt *outerLoop = nullptr;
  auto prove =
      [&](const AdapterBufferEvidence::Access &access) {
        const clang::CallExpr *call = access.call;
        if (call == nullptr || call->getNumArgs() != 3 ||
            !canonicalAdapterPackArgument(
                call, shape, context))
          return false;
        const Expr *rowExpression =
            canonicalAliasExpression(
                call->getArg(1), context);
        ReferencedVariableSet rowReferences;
        rowReferences.TraverseStmt(
            const_cast<Expr *>(rowExpression));
        const VarDecl *rowInGroup = nullptr;
        const VarDecl *outerRow = nullptr;
        for (const VarDecl *variable :
             rowReferences.variables) {
          CanonicalForParts parts;
          if (!canonicalForIndex(
                  variable, context, parts))
            continue;
          if (canonicalUnitLoop(
                  enclosingForOf(variable, context),
                  [&](const Expr *initial) {
                    return canonicalInteger(
                        initial, 0, context);
                  },
                  [&](const Expr *bound) {
                    return canonicalDeclReference(
                        bound, shape.rowsPerAccess,
                        context);
                  }))
            rowInGroup = variable;
          else if (canonicalWarpOuterLoop(
                       enclosingForOf(
                           variable, context),
                       shape, context))
            outerRow = variable;
        }
        if (outerRow == nullptr ||
            rowInGroup == nullptr)
          return false;
        const clang::ForStmt *candidateOuter =
            enclosingForOf(outerRow, context);
        const clang::ForStmt *rowLoop =
            enclosingForOf(rowInGroup, context);
        if (!statementNestedWithin(
                call, candidateOuter, context) ||
            !statementNestedWithin(
                call, rowLoop, context) ||
            (outerLoop != nullptr &&
             outerLoop != candidateOuter))
          return false;
        outerLoop = candidateOuter;
        const bool rowExact =
            shape.family ==
                    CanonicalKernelFamily::SoftmaxWarp
                ? canonicalSoftmaxWarpRow(
                      rowExpression, outerRow,
                      rowInGroup, context)
                : canonicalRmsWarpRow(
                      rowExpression, outerRow,
                      rowInGroup, shape, context);
        if (!rowExact)
          return false;
        const Expr *columnExpression =
            canonicalAliasExpression(
                call->getArg(2), context);
        ReferencedVariableSet columnReferences;
        columnReferences.TraverseStmt(
            const_cast<Expr *>(columnExpression));
        const VarDecl *packIndex = nullptr;
        for (const VarDecl *variable :
             columnReferences.variables) {
          if (enclosingForOf(variable, context) != nullptr) {
            if (packIndex != nullptr)
              return false;
            packIndex = variable;
          }
        }
        if (packIndex == nullptr ||
            !canonicalWarpColumn(
                columnExpression, packIndex,
                shape, context))
          return false;
        const clang::ForStmt *packLoop =
            enclosingForOf(packIndex, context);
        const auto zero =
            [&](const Expr *initial) {
              return canonicalInteger(
                  initial, 0, context);
            };
        const auto minimum =
            [&](const Expr *expression) {
              return canonicalBinaryOperands(
                  expression, clang::BO_Div,
                  [&](const Expr *operand) {
                    return canonicalDeclReference(
                        operand,
                        shape.minimumColumnsPerThread,
                        context);
                  },
                  [&](const Expr *operand) {
                    return canonicalDeclReference(
                        operand, shape.pack,
                        context);
                  },
                  context);
            };
        const auto maximum =
            [&](const Expr *expression) {
              return canonicalBinaryOperands(
                  expression, clang::BO_Div,
                  [&](const Expr *operand) {
                    return canonicalDeclReference(
                        operand,
                        shape.columnsPerThread,
                        context);
                  },
                  [&](const Expr *operand) {
                    return canonicalDeclReference(
                        operand, shape.pack,
                        context);
                  },
                  context);
            };
        bool packLoopExact = false;
        if (shape.family ==
            CanonicalKernelFamily::SoftmaxWarp) {
          packLoopExact = canonicalUnitLoop(
              packLoop, zero, maximum);
        } else {
          packLoopExact =
              canonicalUnitLoop(
                  packLoop, zero, minimum) ||
              canonicalUnitLoop(
                  packLoop, minimum, maximum);
        }
        return packLoopExact &&
               statementNestedWithin(
                   call, packLoop, context) &&
               statementNestedWithin(
                   packLoop, rowLoop, context);
      };
  for (const AdapterBufferEvidence::Access &access :
       adapters.loadCalls) {
    if (!prove(access))
      return false;
  }
  for (const AdapterBufferEvidence::Access &access :
       adapters.storeCalls) {
    if (!prove(access))
      return false;
  }
  return outerLoop != nullptr;
}

bool exactSystemDeclarationNamed(
    const clang::CallExpr *call,
    const FunctionDecl *caller,
    llvm::StringRef name,
    ASTContext &context) {
  std::set<const FunctionDecl *> candidates;
  if (!semanticCallCandidateFunctions(
          call, caller, context, candidates) ||
      candidates.empty())
    return false;
  const SourceManager &sourceManager =
      context.getSourceManager();
  for (const FunctionDecl *candidate : candidates) {
    if (candidate->getName() != name ||
        (candidate->getBuiltinID() == 0 &&
         !sourceManager.isInSystemHeader(
             candidate->getLocation())))
      return false;
  }
  return true;
}

bool exactCudaExecutionBuiltinFetch(
    const clang::CallExpr *call,
    ASTContext &context) {
  const auto *member =
      llvm::dyn_cast_or_null<clang::MemberExpr>(
          stripExpr(
              call == nullptr
                  ? nullptr
                  : call->getCallee()));
  const auto *method =
      member == nullptr
          ? nullptr
          : llvm::dyn_cast<
                clang::CXXMethodDecl>(
                member->getMemberDecl());
  const Expr *object =
      member == nullptr
          ? nullptr
          : member->getBase();
  const clang::PseudoObjectExpr *pseudo = nullptr;
  const Stmt *current = call;
  std::set<const Stmt *> active;
  while (current != nullptr &&
         active.insert(current).second) {
    const auto parents = context.getParents(*current);
    const Stmt *next = nullptr;
    for (const clang::DynTypedNode &parent : parents) {
      if (const auto *candidate =
              parent.get<
                  clang::PseudoObjectExpr>()) {
        pseudo = candidate;
        current = nullptr;
        break;
      }
      if (next == nullptr)
        next = parent.get<Stmt>();
    }
    if (current != nullptr)
      current = next;
  }
  ReferencedVariableSet references;
  if (pseudo != nullptr)
    references.TraverseStmt(
        const_cast<clang::PseudoObjectExpr *>(
            pseudo));
  else if (object != nullptr)
    references.TraverseStmt(
        const_cast<Expr *>(object));
  const VarDecl *builtin =
      references.variables.size() == 1
          ? *references.variables.begin()
          : nullptr;
  const CXXRecordDecl *record =
      builtin == nullptr
          ? nullptr
          : builtin->getType()
                .getNonReferenceType()
                .getUnqualifiedType()
                ->getAsCXXRecordDecl();
  const SourceManager &sourceManager =
      context.getSourceManager();
  return
         call != nullptr &&
         (pseudo == nullptr ||
          stripExpr(pseudo->getResultExpr()) ==
              call) &&
         call->getNumArgs() == 0 &&
         !call->getType()->isVoidType() &&
         method != nullptr &&
         method->getName().starts_with(
             "__fetch_builtin_") &&
         builtin != nullptr &&
         cudaExecutionBuiltin(builtin) &&
         record != nullptr &&
         method->getParent()
                 ->getCanonicalDecl() ==
             record->getCanonicalDecl() &&
         (method->getBuiltinID() != 0 ||
          sourceManager.isInSystemHeader(
              method->getLocation())) &&
         sourceManager.isInSystemHeader(
             builtin->getLocation()) &&
         sourceManager.isInSystemHeader(
             record->getLocation());
}

bool canonicalDomainFailure(
    ControlBranch branch,
    const CanonicalKernelShape &shape,
    ASTContext &context) {
  branch.condition = stripExpr(branch.condition);
  if (const auto *cast =
          llvm::dyn_cast_or_null<
              clang::ExplicitCastExpr>(
                  branch.condition)) {
    if (cast->getType()->isBooleanType())
      branch.condition =
          stripExpr(cast->getSubExpr());
  }
  const EffectiveComparison comparison =
      effectiveComparison(branch);
  if (comparison.opcode == clang::BO_Comma)
    return false;
  if (!shape.warp()) {
    const Expr *dividend = nullptr;
    const Expr *divisor = nullptr;
    const Expr *modulo = comparison.left;
    const Expr *zero = comparison.right;
    if (canonicalInteger(
            comparison.left, 0, context)) {
      modulo = comparison.right;
      zero = comparison.left;
    }
    return comparison.opcode == clang::BO_NE &&
           canonicalInteger(zero, 0, context) &&
           canonicalBinary(
               modulo, clang::BO_Rem,
               dividend, divisor, context) &&
           canonicalDeclReference(
               dividend, shape.columns, context) &&
           canonicalDeclReference(
               divisor, shape.pack, context);
  }
  const Expr *maximum = nullptr;
  const Expr *width = nullptr;
  return comparison.opcode == clang::BO_GT &&
         canonicalDeclReference(
             comparison.left, shape.columns,
             context) &&
         canonicalBinary(
             comparison.right, clang::BO_Mul,
             maximum, width, context) &&
         ((canonicalDeclReference(
                maximum,
                shape.columnsPerThread,
                context) &&
           canonicalDeclReference(
                width,
                shape.threadGroupWidth,
                context)) ||
          (canonicalDeclReference(
                width,
                shape.columnsPerThread,
                context) &&
           canonicalDeclReference(
                maximum,
                shape.threadGroupWidth,
                context)));
}

struct CanonicalAssertEvidence {
  std::set<const clang::CallExpr *> calls;
  std::set<const IfStmt *> conditionals;
  std::set<const clang::ConditionalOperator *>
      operators;
};

bool canonicalAssertControl(
    const clang::CallExpr *call,
    const CanonicalKernelShape &shape,
    ASTContext &context,
    const IfStmt *&ifControl,
    const clang::ConditionalOperator *&operatorControl) {
  ifControl = nullptr;
  operatorControl = nullptr;
  const Stmt *current = call;
  std::set<const Stmt *> active;
  while (current != nullptr &&
         active.insert(current).second) {
    const auto parents = context.getParents(*current);
    const Stmt *next = nullptr;
    for (const clang::DynTypedNode &parent : parents) {
      if (const auto *conditional =
              parent.get<IfStmt>()) {
        bool thenBranch = false;
        if (statementNestedWithin(
                call, conditional->getThen(),
                context))
          thenBranch = true;
        else if (!statementNestedWithin(
                     call, conditional->getElse(),
                     context))
          continue;
        if (canonicalDomainFailure(
                {conditional->getCond(),
                 thenBranch},
                shape, context)) {
          ifControl = conditional;
          return true;
        }
      }
      if (const auto *conditional =
              parent.get<
                  clang::ConditionalOperator>()) {
        bool thenBranch = false;
        if (statementNestedWithin(
                call,
                conditional->getTrueExpr(),
                context))
          thenBranch = true;
        else if (!statementNestedWithin(
                     call,
                     conditional->getFalseExpr(),
                     context))
          continue;
        if (canonicalDomainFailure(
                {conditional->getCond(),
                 thenBranch},
                shape, context)) {
          operatorControl = conditional;
          return true;
        }
      }
      if (next == nullptr)
        next = parent.get<Stmt>();
    }
    current = next;
  }
  return false;
}

class CanonicalAssertCollector
    : public clang::RecursiveASTVisitor<
          CanonicalAssertCollector> {
public:
  CanonicalAssertCollector(
      const FunctionDecl *function,
      const CanonicalKernelShape &shape,
      ASTContext &context)
      : function(function), shape(shape),
        context(context) {}

  bool VisitCallExpr(clang::CallExpr *call) {
    if (!exactSystemDeclarationNamed(
            call, function, "__assert_fail",
            context))
      return true;
    const IfStmt *ifControl = nullptr;
    const clang::ConditionalOperator *operatorControl =
        nullptr;
    if (!canonicalAssertControl(
            call, shape, context, ifControl,
            operatorControl)) {
      safe = false;
      return false;
    }
    evidence.calls.insert(call);
    if (ifControl != nullptr)
      evidence.conditionals.insert(ifControl);
    if (operatorControl != nullptr)
      evidence.operators.insert(operatorControl);
    return true;
  }

  const FunctionDecl *function;
  const CanonicalKernelShape &shape;
  ASTContext &context;
  CanonicalAssertEvidence evidence;
  bool safe = true;
};

bool canonicalPaddingCondition(
    const IfStmt *statement,
    const CanonicalKernelShape &shape,
    const AdapterBufferEvidence &adapters,
    ASTContext &context) {
  if (!shape.warp() || statement == nullptr)
    return false;
  const Expr *notPadding = nullptr;
  const Expr *withinColumns = nullptr;
  if (!canonicalBinary(
          statement->getCond(), clang::BO_LOr,
          notPadding, withinColumns, context))
    return false;
  const auto *negation =
      llvm::dyn_cast_or_null<clang::UnaryOperator>(
          canonicalAliasExpression(
              notPadding, context));
  if (negation == nullptr ||
      negation->getOpcode() != clang::UO_LNot ||
      !canonicalDeclReference(
          negation->getSubExpr(),
          shape.padding, context))
    return false;
  const Expr *column = nullptr;
  const Expr *bound = nullptr;
  if (!canonicalBinary(
          withinColumns, clang::BO_LT,
          column, bound, context) ||
      !canonicalDeclReference(
          bound, shape.columns, context))
    return false;
  const Expr *resolvedColumn =
      canonicalAliasExpression(column, context);
  ReferencedVariableSet references;
  references.TraverseStmt(
      const_cast<Expr *>(resolvedColumn));
  const VarDecl *packIndex = nullptr;
  for (const VarDecl *variable :
       references.variables) {
    if (enclosingForOf(variable, context) == nullptr)
      continue;
    if (packIndex != nullptr)
      return false;
    packIndex = variable;
  }
  if (packIndex == nullptr ||
      !canonicalWarpColumn(
          resolvedColumn, packIndex,
          shape, context))
    return false;
  bool containsAccess = false;
  for (const AdapterBufferEvidence::Access &access :
       adapters.loadCalls)
    containsAccess =
        containsAccess ||
        statementNestedWithin(
            access.call, statement, context);
  for (const AdapterBufferEvidence::Access &access :
       adapters.storeCalls)
    containsAccess =
        containsAccess ||
        statementNestedWithin(
            access.call, statement, context);
  return containsAccess &&
         statementNestedWithin(
             statement,
             enclosingForOf(packIndex, context),
             context);
}

bool canonicalLeaderIf(
    const IfStmt *statement,
    const CanonicalKernelShape &shape,
    ASTContext &context) {
  if (shape.softmax() || statement == nullptr ||
      statement->getElse() != nullptr ||
      !cudaThreadLaneZeroCondition(
          statement->getCond(), context))
    return false;
  const auto *write =
      llvm::dyn_cast_or_null<BinaryOperator>(
          stripAttributedStmt(
              onlyStatement(statement->getThen())));
  return write != nullptr &&
         write->getOpcode() == clang::BO_Assign &&
         storageRoot(write->getLHS()) ==
             shape.inverseOutput;
}

bool canonicalRmsTailIf(
    const IfStmt *statement,
    const CanonicalKernelShape &shape,
    ASTContext &context) {
  if (shape.family !=
          CanonicalKernelFamily::RmsWarp ||
      statement == nullptr ||
      statement->getElse() != nullptr ||
      !llvm::isa_and_nonnull<clang::ContinueStmt>(
          stripAttributedStmt(
              onlyStatement(statement->getThen()))))
    return false;
  const Expr *rowExpression = nullptr;
  const Expr *rows = nullptr;
  if (!canonicalBinary(
          statement->getCond(), clang::BO_GE,
          rowExpression, rows, context) ||
      !canonicalDeclReference(
          rows, shape.rows, context))
    return false;
  rowExpression =
      canonicalAliasExpression(
          rowExpression, context);
  ReferencedVariableSet references;
  references.TraverseStmt(
      const_cast<Expr *>(rowExpression));
  const VarDecl *outerRow = nullptr;
  const VarDecl *rowInGroup = nullptr;
  for (const VarDecl *variable :
       references.variables) {
    const clang::ForStmt *loop =
        enclosingForOf(variable, context);
    if (canonicalWarpOuterLoop(
            loop, shape, context))
      outerRow = variable;
    else if (canonicalUnitLoop(
                 loop,
                 [&](const Expr *initial) {
                   return canonicalInteger(
                       initial, 0, context);
                 },
                 [&](const Expr *bound) {
                   return canonicalDeclReference(
                       bound, shape.rowsPerAccess,
                       context);
                 }))
      rowInGroup = variable;
  }
  return outerRow != nullptr &&
         rowInGroup != nullptr &&
         canonicalRmsWarpRow(
             rowExpression, outerRow,
             rowInGroup, shape, context) &&
         statementNestedWithin(
             statement,
             enclosingForOf(outerRow, context),
             context) &&
         statementNestedWithin(
             statement,
             enclosingForOf(rowInGroup, context),
             context);
}

class CanonicalKernelControlVisitor
    : public clang::RecursiveASTVisitor<
          CanonicalKernelControlVisitor> {
public:
  CanonicalKernelControlVisitor(
      const CanonicalKernelShape &shape,
      const AdapterBufferEvidence &adapters,
      const CanonicalAssertEvidence &assertions,
      ASTContext &context)
      : shape(shape), adapters(adapters),
        assertions(assertions), context(context) {}

  bool VisitIfStmt(IfStmt *statement) {
    if (assertions.conditionals.count(statement) != 0)
      return true;
    const NonTypeTemplateParmDecl *parameter = nullptr;
    const EnumConstantDecl *constant = nullptr;
    if (shape.softmax() &&
        enumEquality(
            statement->getCond(), parameter,
            constant) &&
        parameter == shape.algorithm)
      return true;
    if (canonicalPaddingCondition(
            statement, shape, adapters,
            context))
      return true;
    if (canonicalLeaderIf(
            statement, shape, context))
      return true;
    if (canonicalRmsTailIf(
            statement, shape, context)) {
      tailConditionals.insert(statement);
      return true;
    }
    safe = false;
    return false;
  }

  bool VisitContinueStmt(
      clang::ContinueStmt *statement) {
    for (const IfStmt *conditional :
         tailConditionals) {
      if (statementNestedWithin(
              statement, conditional, context))
        return true;
    }
    safe = false;
    return false;
  }

  bool VisitBreakStmt(clang::BreakStmt *) {
    safe = false;
    return false;
  }

  bool VisitWhileStmt(clang::WhileStmt *) {
    safe = false;
    return false;
  }

  bool VisitDoStmt(clang::DoStmt *) {
    safe = false;
    return false;
  }

  bool VisitSwitchStmt(clang::SwitchStmt *) {
    safe = false;
    return false;
  }

  bool VisitConditionalOperator(
      clang::ConditionalOperator *operation) {
    if (assertions.operators.count(operation) == 0)
      safe = false;
    return safe;
  }

  bool TraverseLambdaExpr(
      clang::LambdaExpr *) {
    safe = false;
    return false;
  }

  const CanonicalKernelShape &shape;
  const AdapterBufferEvidence &adapters;
  const CanonicalAssertEvidence &assertions;
  ASTContext &context;
  std::set<const IfStmt *> tailConditionals;
  bool safe = true;
};

bool canonicalKernelCoverageAndControl(
    const FunctionDecl *function,
    bool softmax,
    const AdapterBufferEvidence &adapters,
    ASTContext &context,
    std::set<const clang::CallExpr *>
        &allowedStructuralCalls) {
  const CanonicalKernelShape shape =
      canonicalKernelShape(function, softmax);
  if (shape.family == CanonicalKernelFamily::None)
    return false;
  const bool accesses =
      shape.warp()
          ? canonicalWarpAdapterAccesses(
                adapters, shape, context)
          : canonicalBlockAdapterAccesses(
                adapters, shape, context);
  if (!accesses)
    return false;
  CanonicalAssertCollector assertions(
      function, shape, context);
  assertions.TraverseStmt(
      const_cast<Stmt *>(function->getBody()));
  if (!assertions.safe ||
      assertions.evidence.calls.size() != 1)
    return false;
  CanonicalKernelControlVisitor controls(
      shape, adapters, assertions.evidence,
      context);
  controls.TraverseStmt(
      const_cast<Stmt *>(function->getBody()));
  const unsigned expectedTail =
      shape.family ==
              CanonicalKernelFamily::RmsWarp
          ? 2
          : 0;
  if (!controls.safe ||
      controls.tailConditionals.size() !=
          expectedTail)
    return false;
  allowedStructuralCalls.insert(
      assertions.evidence.calls.begin(),
      assertions.evidence.calls.end());
  return true;
}

bool comparisonsConflict(
    ControlBranch lhs,
    ControlBranch rhs,
    ASTContext &context) {
  const EffectiveComparison left =
      effectiveComparison(lhs);
  const EffectiveComparison right =
      effectiveComparison(rhs);
  if (left.opcode == clang::BO_Comma ||
      right.opcode == clang::BO_Comma)
    return false;
  AlphaExpressionContext operands{context};
  const bool sameLeft =
      alphaEquivalentExpression(
          left.left, right.left, operands);
  AlphaExpressionContext values{context};
  const bool sameRight =
      alphaEquivalentExpression(
          left.right, right.right, values);
  if (sameLeft && sameRight &&
      invertedComparison(left.opcode) ==
          right.opcode)
    return true;
  if (!sameLeft ||
      left.opcode != clang::BO_EQ ||
      right.opcode != clang::BO_EQ)
    return false;
  Expr::EvalResult leftValue;
  Expr::EvalResult rightValue;
  return left.right->EvaluateAsInt(
             leftValue, context) &&
         right.right->EvaluateAsInt(
             rightValue, context) &&
         leftValue.Val.isInt() &&
         rightValue.Val.isInt() &&
         leftValue.Val.getInt() !=
             rightValue.Val.getInt();
}

bool controlContextsCompatible(
    const std::vector<const Stmt *> &statements,
    ASTContext &context,
    const SoftmaxChoiceKey *choice = nullptr) {
  std::vector<std::vector<ControlBranch>> contexts;
  for (const Stmt *statement : statements) {
    if (statement == nullptr ||
        discardedOrDeferredExecution(
            statement, context))
      return false;
    std::vector<ControlBranch> branches =
        enclosingControlBranches(statement, context);
    std::vector<ControlBranch> filtered;
    for (ControlBranch branch : branches) {
      const NonTypeTemplateParmDecl *parameter = nullptr;
      const EnumConstantDecl *constant = nullptr;
      if (choice != nullptr &&
          enumEquality(
              branch.condition, parameter, constant) &&
          parameter == choice->parameter)
        continue;
      filtered.push_back(branch);
    }
    contexts.push_back(std::move(filtered));
  }
  for (const std::vector<ControlBranch> &branches :
       contexts) {
    for (unsigned lhs = 0; lhs < branches.size(); ++lhs) {
      for (unsigned rhs = lhs + 1;
           rhs < branches.size(); ++rhs) {
        if (comparisonsConflict(
                branches[lhs], branches[rhs],
                context))
          return false;
      }
    }
  }
  for (unsigned lhsContext = 0;
       lhsContext < contexts.size(); ++lhsContext) {
    for (unsigned rhsContext = lhsContext + 1;
         rhsContext < contexts.size(); ++rhsContext) {
      if (contexts[lhsContext].empty() ||
          contexts[rhsContext].empty())
        continue;
      bool sharedRequirement = false;
      for (const ControlBranch &lhs :
           contexts[lhsContext]) {
        for (const ControlBranch &rhs :
             contexts[rhsContext]) {
          if (comparisonsConflict(lhs, rhs, context))
            return false;
          const ControlBranch normalizedLhs =
              normalizedControlRequirement(
                  lhs, context);
          const ControlBranch normalizedRhs =
              normalizedControlRequirement(
                  rhs, context);
          AlphaExpressionContext alpha{context};
          if (!alphaEquivalentExpression(
                  normalizedLhs.condition,
                  normalizedRhs.condition, alpha))
            continue;
          sharedRequirement = true;
          if (normalizedLhs.thenBranch !=
              normalizedRhs.thenBranch)
            return false;
        }
      }
      if (!sharedRequirement)
        return false;
    }
  }
  return true;
}

bool adapterCallShapesEquivalent(
    const clang::CallExpr *lhs,
    const clang::CallExpr *rhs,
    ASTContext &context) {
  if (lhs == nullptr || rhs == nullptr ||
      lhs->getNumArgs() != 3 ||
      rhs->getNumArgs() != 3)
    return false;
  const std::vector<clang::TemplateArgument> lhsTemplates =
      memberTemplateArguments(lhs);
  const std::vector<clang::TemplateArgument> rhsTemplates =
      memberTemplateArguments(rhs);
  if (lhsTemplates.size() != 1 ||
      rhsTemplates.size() != 1 ||
      lhsTemplates[0].getKind() !=
          clang::TemplateArgument::Expression ||
      rhsTemplates[0].getKind() !=
          clang::TemplateArgument::Expression)
    return false;
  AlphaExpressionContext alpha{context};
  if (!alphaEquivalentExpression(
          lhsTemplates[0].getAsExpr(),
          rhsTemplates[0].getAsExpr(), alpha) ||
      !alphaEquivalentExpression(
          lhs->getArg(1), rhs->getArg(1), alpha) ||
      !alphaEquivalentExpression(
          lhs->getArg(2), rhs->getArg(2), alpha))
    return false;
  const std::vector<ControlBranch> lhsBranches =
      enclosingControlBranches(lhs, context);
  const std::vector<ControlBranch> rhsBranches =
      enclosingControlBranches(rhs, context);
  if (lhsBranches.size() != rhsBranches.size())
    return false;
  for (unsigned index = 0;
       index < lhsBranches.size(); ++index) {
    if (lhsBranches[index].thenBranch !=
            rhsBranches[index].thenBranch ||
        !alphaEquivalentExpression(
            lhsBranches[index].condition,
            rhsBranches[index].condition, alpha))
      return false;
  }
  return true;
}

bool equivalentAdapterLoads(
    const AdapterBufferEvidence &adapters,
    const Decl *lhsRoot,
    const Decl *rhsRoot,
    ASTContext &context) {
  for (const AdapterBufferEvidence::Access &lhs :
       adapters.loadCalls) {
    if (lhs.root != lhsRoot)
      continue;
    for (const AdapterBufferEvidence::Access &rhs :
         adapters.loadCalls) {
      if (rhs.root == rhsRoot &&
          adapterCallShapesEquivalent(
              lhs.call, rhs.call, context))
        return true;
    }
  }
  return false;
}

bool storageCopiesReachBefore(
    const Stmt *body,
    const Decl *from,
    const Decl *to,
    const Stmt *operation,
    SourceManager &sourceManager) {
  if (from == nullptr || to == nullptr ||
      operation == nullptr)
    return false;
  if (from == to)
    return true;
  StorageCopyCollector copies;
  copies.TraverseStmt(const_cast<Stmt *>(body));
  std::sort(
      copies.edges.begin(), copies.edges.end(),
      [&](const StorageCopyEdge &lhs,
          const StorageCopyEdge &rhs) {
        return sourceManager.isBeforeInTranslationUnit(
            sourceManager.getExpansionLoc(lhs.location),
            sourceManager.getExpansionLoc(rhs.location));
      });
  const SourceLocation end =
      sourceManager.getExpansionLoc(
          operation->getBeginLoc());
  std::set<const Decl *> reachable = {from};
  for (const StorageCopyEdge &edge : copies.edges) {
    const SourceLocation location =
        sourceManager.getExpansionLoc(edge.location);
    if (!sourceManager.isBeforeInTranslationUnit(
            location, end))
      continue;
    if (reachable.count(edge.from) != 0)
      reachable.insert(edge.to);
  }
  return reachable.count(to) != 0;
}

bool adapterLoadReachesBefore(
    const Stmt *body,
    const AdapterBufferEvidence &adapters,
    const Decl *target,
    const Stmt *operation,
    SourceManager &sourceManager) {
  if (body == nullptr || target == nullptr ||
      operation == nullptr)
    return false;
  StorageCopyCollector copies;
  copies.TraverseStmt(const_cast<Stmt *>(body));
  std::sort(
      copies.edges.begin(), copies.edges.end(),
      [&](const StorageCopyEdge &lhs,
          const StorageCopyEdge &rhs) {
        return sourceManager.isBeforeInTranslationUnit(
            sourceManager.getExpansionLoc(lhs.location),
            sourceManager.getExpansionLoc(rhs.location));
      });
  const SourceLocation end =
      sourceManager.getExpansionLoc(
          operation->getBeginLoc());
  for (const AdapterBufferEvidence::Access &load :
       adapters.loadCalls) {
    if (load.root == nullptr || load.call == nullptr)
      continue;
    const SourceLocation begin =
        sourceManager.getExpansionLoc(
            load.call->getBeginLoc());
    if (!sourceManager.isBeforeInTranslationUnit(begin, end))
      continue;
    std::set<const Decl *> reachable = {load.root};
    for (const StorageCopyEdge &edge : copies.edges) {
      const SourceLocation location =
          sourceManager.getExpansionLoc(edge.location);
      if (!sourceManager.isBeforeInTranslationUnit(
              begin, location) ||
          !sourceManager.isBeforeInTranslationUnit(
              location, end))
        continue;
      if (reachable.count(edge.from) != 0)
        reachable.insert(edge.to);
    }
    if (reachable.count(target) != 0)
      return true;
  }
  return false;
}

bool adapterAccessReachesBefore(
    const Stmt *body,
    const AdapterBufferEvidence::Access &load,
    const Decl *target,
    const Stmt *operation,
    SourceManager &sourceManager) {
  if (body == nullptr || load.root == nullptr ||
      load.call == nullptr || target == nullptr ||
      operation == nullptr)
    return false;
  const SourceLocation begin =
      sourceManager.getExpansionLoc(
          load.call->getBeginLoc());
  const SourceLocation end =
      sourceManager.getExpansionLoc(
          operation->getBeginLoc());
  if (!sourceManager.isBeforeInTranslationUnit(
          begin, end))
    return false;
  std::set<const Decl *> reachable = {load.root};
  StorageCopyCollector copies;
  copies.TraverseStmt(const_cast<Stmt *>(body));
  std::sort(
      copies.edges.begin(), copies.edges.end(),
      [&](const StorageCopyEdge &lhs,
          const StorageCopyEdge &rhs) {
        return sourceManager.isBeforeInTranslationUnit(
            sourceManager.getExpansionLoc(lhs.location),
            sourceManager.getExpansionLoc(rhs.location));
      });
  for (const StorageCopyEdge &edge : copies.edges) {
    const SourceLocation location =
        sourceManager.getExpansionLoc(edge.location);
    if (!sourceManager.isBeforeInTranslationUnit(
            begin, location) ||
        !sourceManager.isBeforeInTranslationUnit(
            location, end))
      continue;
    if (reachable.count(edge.from) != 0)
      reachable.insert(edge.to);
  }
  return reachable.count(target) != 0;
}

bool normalizationReachesStore(
    const Stmt *body,
    const Decl *root,
    const BinaryOperator *normalization,
    const std::vector<AdapterBufferEvidence::Access> &stores,
    SourceManager &sourceManager,
    const SoftmaxChoiceKey *choice);

bool normalizationReachesSpecificStore(
    const Stmt *body,
    const Decl *root,
    const BinaryOperator *normalization,
    const AdapterBufferEvidence::Access &store,
    SourceManager &sourceManager,
    const SoftmaxChoiceKey *choice,
    const std::set<const clang::CallExpr *>
        *allowedReadCalls = nullptr);

bool softmaxPipelineExecutable(
    const FunctionDecl *function,
    const SoftmaxExpEvidence &exponential,
    const SoftmaxNormalizeEvidence &normalization,
    bool materializedFlow,
    const AdapterBufferEvidence &adapters,
    const std::set<const clang::CallExpr *>
        &reductionCalls,
    ASTContext &context,
    SourceManager &sourceManager,
    const SoftmaxChoiceKey &choice) {
  std::vector<
      const AdapterBufferEvidence::Access *>
      normalizationLoads;
  if (materializedFlow) {
    normalizationLoads.push_back(nullptr);
  } else {
    for (const AdapterBufferEvidence::Access &load :
         adapters.loadCalls) {
      if (adapterAccessReachesBefore(
              function->getBody(), load,
              normalization.inlineExpRoot,
              normalization.operation,
              sourceManager))
        normalizationLoads.push_back(&load);
    }
  }
  if (normalizationLoads.empty())
    return false;
  for (const AdapterBufferEvidence::Access &load :
       adapters.loadCalls) {
    if (!adapterAccessReachesBefore(
            function->getBody(), load,
            exponential.bufferRoot,
            exponential.operation,
            sourceManager))
      continue;
    for (const AdapterBufferEvidence::Access
             *normalizationLoad : normalizationLoads) {
      for (const AdapterBufferEvidence::Access &store :
           adapters.storeCalls) {
        if (!normalizationReachesSpecificStore(
                function->getBody(),
                normalization.outputRoot,
                normalization.operation, store,
                sourceManager, &choice))
          continue;
        std::vector<const Stmt *> operations = {
            load.call, exponential.operation,
            exponential.accumulationOperation,
            normalization.operation, store.call};
        for (const clang::CallExpr *reduction :
             reductionCalls)
          operations.push_back(reduction);
        if (normalizationLoad != nullptr)
          operations.push_back(
              normalizationLoad->call);
        if (controlContextsCompatible(
                operations, context, &choice))
          return true;
      }
    }
  }
  return false;
}

bool softmaxAdapterAccessTopologyExact(
    const FunctionDecl *function,
    const SoftmaxExpEvidence &exponential,
    const SoftmaxNormalizeEvidence &normalization,
    bool materializedFlow,
    const AdapterBufferEvidence &adapters,
    ASTContext &context,
    SourceManager &sourceManager,
    const SoftmaxChoiceKey &choice) {
  const bool exactCounts =
      materializedFlow
          ? (adapters.loadCalls.size() == 1 &&
             adapters.storeCalls.size() == 1)
          : (adapters.loadCalls.size() == 3 &&
             adapters.storeCalls.size() == 1);
  if (!exactCounts)
    return false;
  if (!materializedFlow) {
    for (unsigned lhs = 0;
         lhs < adapters.loadCalls.size(); ++lhs) {
      for (unsigned rhs = lhs + 1;
           rhs < adapters.loadCalls.size(); ++rhs) {
        if (!adapterCallShapesEquivalent(
                adapters.loadCalls[lhs].call,
                adapters.loadCalls[rhs].call,
                context))
          return false;
      }
    }
  }
  unsigned relevantLoads = 0;
  for (const AdapterBufferEvidence::Access &load :
       adapters.loadCalls) {
    const bool feedsExponential =
        adapterAccessReachesBefore(
            function->getBody(), load,
            exponential.bufferRoot,
            exponential.operation, sourceManager);
    const bool feedsNormalization =
        !materializedFlow &&
        adapterAccessReachesBefore(
            function->getBody(), load,
            normalization.inlineExpRoot,
            normalization.operation, sourceManager);
    if (feedsExponential || feedsNormalization)
      ++relevantLoads;
  }
  if ((materializedFlow && relevantLoads != 1) ||
      (!materializedFlow && relevantLoads != 2))
    return false;
  for (const AdapterBufferEvidence::Access &store :
       adapters.storeCalls) {
    if (!normalizationReachesSpecificStore(
            function->getBody(),
            normalization.outputRoot,
            normalization.operation, store,
            sourceManager, &choice))
      return false;
  }
  return true;
}

void collectSoftmaxChoices(
    const FunctionDecl *function,
    ASTContext &context,
    PrimitiveRegistry &registry,
    SourceManager &sourceManager,
    std::set<const EnumConstantDecl *> &choices,
    std::set<const FunctionDecl *> &kernels,
    SoftmaxKernelBindings &bindings) {
  if (function == nullptr || !function->hasBody() ||
      !function->hasAttr<clang::CUDAGlobalAttr>() ||
      function->getNumParams() != 4 ||
      !templateTypeParameter(
          function->getParamDecl(0)->getType(),
          0, 0, false, false, false) ||
      !templateTypeParameter(
          function->getParamDecl(1)->getType(),
          0, 1, false, false, false) ||
      !isIntegralWidthAtLeast(
          function->getParamDecl(2)->getType(), context, 32) ||
      !isIntegralWidthAtLeast(
          function->getParamDecl(3)->getType(), context, 32))
    return;

  AdapterCallVisitor adapterCalls(
      function->getParamDecl(0), function->getParamDecl(1),
      context);
  adapterCalls.TraverseStmt(
      const_cast<Stmt *>(function->getBody()));
  if (!adapterCalls.proven())
    return;
  AdapterBufferVisitor adapterBuffers(
      function->getParamDecl(0),
      function->getParamDecl(1), context);
  adapterBuffers.TraverseStmt(
      const_cast<Stmt *>(function->getBody()));
  std::set<const clang::CallExpr *>
      allowedStructuralCalls;
  if (!canonicalKernelCoverageAndControl(
          function, true,
          adapterBuffers.evidence, context,
          allowedStructuralCalls))
    return;
  std::set<const clang::CallExpr *> allowedAdapterCalls;
  for (const AdapterBufferEvidence::Access &access :
       adapterBuffers.evidence.loadCalls)
    allowedAdapterCalls.insert(access.call);
  for (const AdapterBufferEvidence::Access &access :
       adapterBuffers.evidence.storeCalls)
    allowedAdapterCalls.insert(access.call);

  SoftmaxKernelVisitor visitor(registry, context);
  visitor.TraverseStmt(
      const_cast<Stmt *>(function->getBody()));
  const SoftmaxBranchEvidence shared =
      analyzeSoftmaxBranch(
          function->getBody(), registry, context);
  EnumUncontrolledExpCollector uncontrolled(registry);
  uncontrolled.TraverseStmt(
      const_cast<Stmt *>(function->getBody()));
  for (const SoftmaxExpEvidence &exponential :
       shared.exponentials) {
    const auto *call =
        llvm::dyn_cast_or_null<clang::CallExpr>(
            exponential.operation);
    if (call == nullptr ||
        uncontrolled.calls.count(call) == 0)
      continue;
    for (auto &entry : visitor.evidence)
      entry.second.exponentials.push_back(
          exponential);
  }
  for (const auto &entry : visitor.evidence) {
    for (const SoftmaxExpEvidence &exponential :
         entry.second.exponentials) {
      for (const SoftmaxNormalizeEvidence &normalization :
           entry.second.normalizations) {
        const bool materializedFlow =
            exponential.materializedRoot != nullptr &&
            exponential.materializedRoot ==
                normalization.sourceRoot;
        const bool inlineFlow =
            exponential.materializedRoot == nullptr &&
            normalization.inlineExpRoot != nullptr &&
            (normalization.inlineExpRoot ==
                 exponential.bufferRoot ||
             equivalentAdapterLoads(
                 adapterBuffers.evidence,
                 exponential.bufferRoot,
                 normalization.inlineExpRoot,
                 context)) &&
            normalization.shiftRoot ==
                exponential.shiftRoot;
        const bool stored =
            adapterBuffers.evidence.stored.count(
                normalization.outputRoot) != 0;
        const bool loaded =
            adapterLoadReachesBefore(
                function->getBody(),
                adapterBuffers.evidence,
                exponential.bufferRoot,
                exponential.operation,
                sourceManager);
        const bool normalizationLoaded =
            materializedFlow ||
            (normalization.inlineExpRoot != nullptr &&
             adapterLoadReachesBefore(
                 function->getBody(),
                 adapterBuffers.evidence,
                 normalization.inlineExpRoot,
                 normalization.operation,
                 sourceManager));
        const bool reachesStore =
            normalizationReachesStore(
                function->getBody(),
                normalization.outputRoot,
                normalization.operation,
                adapterBuffers.evidence.storeCalls,
                sourceManager, &entry.first);
        std::set<const clang::CallExpr *>
            allowedSemanticCalls =
                allowedStructuralCalls;
        const bool maximumFlow =
            storageProducedByReduction(
                function->getBody(),
                exponential.shiftRoot,
                exponential.operation,
                ProvenReductionKind::Maximum,
                function, context,
                &allowedSemanticCalls);
        const bool sumFlow =
            storageFlowsThroughCall(
                function->getBody(),
                exponential.sumRoot,
                normalization.sumRoot,
                normalization.operation,
                function, context,
                &allowedSemanticCalls);
        const bool executablePipeline =
            softmaxPipelineExecutable(
                function, exponential, normalization,
                materializedFlow,
                adapterBuffers.evidence,
                allowedSemanticCalls, context,
                sourceManager, entry.first);
        const bool exactAccessTopology =
            softmaxAdapterAccessTopologyExact(
                function, exponential, normalization,
                materializedFlow,
                adapterBuffers.evidence,
                context,
                sourceManager, entry.first);
        if ((materializedFlow || inlineFlow) &&
            stored && loaded && normalizationLoaded &&
            reachesStore &&
            maximumFlow && sumFlow &&
            executablePipeline &&
            exactAccessTopology &&
            semanticEffectsAllowed(
                function, registry,
                allowedAdapterCalls,
                allowedSemanticCalls, nullptr,
                nullptr, &entry.first, context)) {
          choices.insert(entry.first.constant);
          const FunctionDecl *kernel =
              function->getCanonicalDecl();
          kernels.insert(kernel);
          bindings[kernel].insert(entry.first);
        }
      }
    }
  }
}

class EnumReferenceVisitor
    : public clang::RecursiveASTVisitor<EnumReferenceVisitor> {
public:
  explicit EnumReferenceVisitor(
      const std::set<const EnumConstantDecl *> &wanted)
      : wanted(wanted) {}

  bool VisitDeclRefExpr(DeclRefExpr *reference) {
    const auto *constant =
        llvm::dyn_cast<EnumConstantDecl>(reference->getDecl());
    if (constant != nullptr && wanted.count(constant) != 0)
      found = true;
    return !found;
  }

  bool contains(const Stmt *statement) {
    found = false;
    TraverseStmt(const_cast<Stmt *>(statement));
    return found;
  }

private:
  const std::set<const EnumConstantDecl *> &wanted;
  bool found = false;
};

bool callForwardsParameters(const clang::CallExpr *call,
                            const FunctionDecl *function,
                            unsigned count) {
  if (call == nullptr || function == nullptr ||
      call->getNumArgs() < count ||
      function->getNumParams() < count)
    return false;
  for (unsigned index = 0; index < count; ++index) {
    if (directReferencedAs<ParmVarDecl>(
            call->getArg(index)) !=
        function->getParamDecl(index))
      return false;
  }
  return true;
}

using FunctionGraph =
    std::map<const FunctionDecl *,
             std::set<const FunctionDecl *>>;

bool hasRowWidthRoutingShape(const FunctionDecl *function,
                             ASTContext &context,
                             unsigned widthParameterIndex,
                             unsigned forwardedParameterCount);

bool callCandidateFunctions(
    const clang::CallExpr *call,
    const FunctionDecl *caller,
    std::set<const FunctionDecl *> &functions) {
  if (call == nullptr)
    return false;
  if (const FunctionDecl *direct = call->getDirectCallee()) {
    functions.insert(direct->getCanonicalDecl());
    return true;
  }
  const Expr *callee = stripExpr(call->getCallee());
  if (const auto *reference =
          llvm::dyn_cast_or_null<DeclRefExpr>(callee)) {
    const FunctionDecl *function =
        functionFromNamedDecl(reference->getDecl());
    if (function == nullptr)
      return false;
    functions.insert(function->getCanonicalDecl());
    return true;
  }
  const auto *unresolved =
      llvm::dyn_cast_or_null<clang::UnresolvedLookupExpr>(callee);
  if (unresolved == nullptr || unresolved->requiresADL())
    return false;
  for (const NamedDecl *declaration : unresolved->decls()) {
    const FunctionDecl *function =
        functionFromNamedDecl(declaration);
    if (function == nullptr)
      return false;
    functions.insert(function->getCanonicalDecl());
  }
  (void)caller;
  return !functions.empty();
}

bool unresolvedFunctorCandidates(
    const clang::CallExpr *call,
    std::set<const FunctionDecl *> &functions) {
  const auto *construction =
      llvm::dyn_cast_or_null<clang::CXXUnresolvedConstructExpr>(
          stripExpr(call == nullptr ? nullptr : call->getCallee()));
  if (construction == nullptr)
    return false;
  const auto *specialization =
      construction->getType()
          .getNonReferenceType()
          .getUnqualifiedType()
          ->getAs<clang::TemplateSpecializationType>();
  const clang::TemplateDecl *templateDeclaration =
      specialization == nullptr
          ? nullptr
          : specialization->getTemplateName()
                .getAsTemplateDecl();
  const auto *classTemplate =
      llvm::dyn_cast_or_null<clang::ClassTemplateDecl>(
          templateDeclaration);
  const CXXRecordDecl *record =
      classTemplate == nullptr
          ? nullptr
          : classTemplate->getTemplatedDecl();
  if (record == nullptr)
    return false;
  for (const clang::CXXMethodDecl *method : record->methods()) {
    if (method->getOverloadedOperator() ==
        clang::OO_Call)
      functions.insert(method->getCanonicalDecl());
  }
  return !functions.empty();
}

struct RouteBinding {
  std::set<const TemplateTypeParmDecl *> computeTypes;
  std::map<const NonTypeTemplateParmDecl *,
           const EnumConstantDecl *> enums;
};

const TemplateTypeParmDecl *referencedTypeParameter(
    QualType type) {
  type = type.getNonReferenceType().getUnqualifiedType();
  const auto *parameter =
      type->getAs<clang::TemplateTypeParmType>();
  return parameter == nullptr ? nullptr : parameter->getDecl();
}

const EnumConstantDecl *resolveEnumArgument(
    const clang::TemplateArgument &argument,
    const RouteBinding &binding) {
  if (argument.getKind() ==
      clang::TemplateArgument::Declaration)
    return llvm::dyn_cast<EnumConstantDecl>(
        argument.getAsDecl());
  if (argument.getKind() !=
      clang::TemplateArgument::Expression)
    return nullptr;
  const Expr *expression = argument.getAsExpr();
  if (const auto *constant =
          directReferencedAs<EnumConstantDecl>(expression))
    return constant;
  const auto *parameter =
      directReferencedAs<NonTypeTemplateParmDecl>(expression);
  const auto found = binding.enums.find(parameter);
  return found == binding.enums.end()
             ? nullptr
             : found->second;
}

std::vector<clang::TemplateArgument>
callTemplateArguments(const clang::CallExpr *call,
                      const FunctionDecl *candidate) {
  std::vector<clang::TemplateArgument> arguments;
  const Expr *callee = stripExpr(call->getCallee());
  if (const auto *unresolved =
          llvm::dyn_cast_or_null<
              clang::UnresolvedLookupExpr>(callee)) {
    if (unresolved->hasExplicitTemplateArgs()) {
      for (const clang::TemplateArgumentLoc &argument :
           unresolved->template_arguments())
        arguments.push_back(argument.getArgument());
    }
  } else if (const auto *reference =
                 llvm::dyn_cast_or_null<DeclRefExpr>(callee)) {
    if (reference->hasExplicitTemplateArgs()) {
      for (const clang::TemplateArgumentLoc &argument :
           reference->template_arguments())
        arguments.push_back(argument.getArgument());
    }
  }
  if (!arguments.empty())
    return arguments;
  if (const clang::TemplateArgumentList *specialization =
          candidate->getTemplateSpecializationArgs()) {
    for (const clang::TemplateArgument &argument :
         specialization->asArray())
      arguments.push_back(argument);
  }
  return arguments;
}

const FunctionTemplateDecl *primaryFunctionTemplate(
    const FunctionDecl *function) {
  if (function == nullptr)
    return nullptr;
  if (const FunctionTemplateDecl *primary =
          function->getPrimaryTemplate())
    return primary;
  return function->getDescribedFunctionTemplate();
}

void bindTemplateArguments(
    const TemplateParameterList *parameters,
    const std::vector<clang::TemplateArgument> &arguments,
    const RouteBinding &source,
    RouteBinding &destination) {
  if (parameters == nullptr)
    return;
  const unsigned count =
      std::min<unsigned>(
          parameters->size(), arguments.size());
  for (unsigned index = 0; index < count; ++index) {
    const Decl *parameter = parameters->getParam(index);
    const clang::TemplateArgument &argument =
        arguments[index];
    if (const auto *typeParameter =
            llvm::dyn_cast<TemplateTypeParmDecl>(parameter)) {
      if (argument.getKind() !=
          clang::TemplateArgument::Type)
        continue;
      const TemplateTypeParmDecl *sourceParameter =
          referencedTypeParameter(argument.getAsType());
      if (source.computeTypes.count(sourceParameter) != 0)
        destination.computeTypes.insert(typeParameter);
      continue;
    }
    const auto *valueParameter =
        llvm::dyn_cast<NonTypeTemplateParmDecl>(parameter);
    if (valueParameter == nullptr ||
        !valueParameter->getType()->isEnumeralType())
      continue;
    if (const EnumConstantDecl *constant =
            resolveEnumArgument(argument, source))
      destination.enums[valueParameter] = constant;
  }
}

const Expr *callObject(const clang::CallExpr *call) {
  if (const auto *member =
          llvm::dyn_cast<clang::CXXMemberCallExpr>(call))
    return member->getImplicitObjectArgument();
  if (const auto *operation =
          llvm::dyn_cast<clang::CXXOperatorCallExpr>(call)) {
    if (operation->getNumArgs() != 0)
      return operation->getArg(0);
  }
  if (llvm::isa_and_nonnull<clang::CXXUnresolvedConstructExpr>(
          stripExpr(call == nullptr ? nullptr : call->getCallee())))
    return stripExpr(call->getCallee());
  return nullptr;
}

RouteBinding bindRouteCall(
    const clang::CallExpr *call,
    const FunctionDecl *candidate,
    const RouteBinding &source) {
  RouteBinding result;
  const FunctionTemplateDecl *functionTemplate =
      primaryFunctionTemplate(candidate);
  if (functionTemplate != nullptr) {
    bindTemplateArguments(
        functionTemplate->getTemplateParameters(),
        callTemplateArguments(call, candidate),
        source, result);
  }
  const auto *method =
      llvm::dyn_cast<clang::CXXMethodDecl>(candidate);
  const Expr *object = callObject(call);
  if (method == nullptr || object == nullptr)
    return result;
  const CXXRecordDecl *parent = method->getParent();
  const auto *classTemplate =
      parent->getDescribedClassTemplate();
  const auto *specialization =
      object->getType()
          .getNonReferenceType()
          .getUnqualifiedType()
          ->getAs<clang::TemplateSpecializationType>();
  if (classTemplate == nullptr || specialization == nullptr)
    return result;
  std::vector<clang::TemplateArgument> arguments(
      specialization->template_arguments().begin(),
      specialization->template_arguments().end());
  bindTemplateArguments(
      classTemplate->getTemplateParameters(),
      arguments, source, result);
  return result;
}

class ExecutableCallCollector
    : public clang::RecursiveASTVisitor<ExecutableCallCollector> {
public:
  struct FoundCall {
    const clang::CallExpr *call = nullptr;
    unsigned conditionalDepth = 0;
  };

  bool VisitCallExpr(clang::CallExpr *call) {
    calls.push_back({call, conditionalDepth});
    return true;
  }

  bool TraverseIfStmt(IfStmt *statement) {
    if (statement == nullptr)
      return true;
    TraverseStmt(statement->getInit());
    TraverseStmt(statement->getConditionVariableDeclStmt());
    TraverseStmt(statement->getCond());
    Expr::EvalResult evaluated;
    if (statement->getCond() != nullptr &&
        statement->getCond()->EvaluateAsInt(
            evaluated, *context) &&
        evaluated.Val.isInt()) {
      if (evaluated.Val.getInt() != 0)
        TraverseStmt(statement->getThen());
      else
        TraverseStmt(statement->getElse());
      return true;
    }
    ++conditionalDepth;
    TraverseStmt(statement->getThen());
    TraverseStmt(statement->getElse());
    --conditionalDepth;
    return true;
  }

  bool TraverseForStmt(clang::ForStmt *statement) {
    if (statement == nullptr)
      return true;
    TraverseStmt(statement->getInit());
    TraverseStmt(statement->getConditionVariableDeclStmt());
    TraverseStmt(statement->getCond());
    Expr::EvalResult evaluated;
    if (statement->getCond() != nullptr &&
        statement->getCond()->EvaluateAsInt(
            evaluated, *context) &&
        evaluated.Val.isInt() &&
        evaluated.Val.getInt() == 0)
      return true;
    ++conditionalDepth;
    TraverseStmt(statement->getBody());
    --conditionalDepth;
    TraverseStmt(statement->getInc());
    return true;
  }

  bool TraverseWhileStmt(clang::WhileStmt *statement) {
    if (statement == nullptr)
      return true;
    TraverseStmt(statement->getConditionVariableDeclStmt());
    TraverseStmt(statement->getCond());
    Expr::EvalResult evaluated;
    if (statement->getCond() != nullptr &&
        statement->getCond()->EvaluateAsInt(
            evaluated, *context) &&
        evaluated.Val.isInt() &&
        evaluated.Val.getInt() == 0)
      return true;
    ++conditionalDepth;
    TraverseStmt(statement->getBody());
    --conditionalDepth;
    return true;
  }

  bool TraverseDoStmt(clang::DoStmt *statement) {
    if (statement == nullptr)
      return true;
    ++conditionalDepth;
    TraverseStmt(statement->getBody());
    --conditionalDepth;
    TraverseStmt(statement->getCond());
    return true;
  }

  bool TraverseSwitchStmt(clang::SwitchStmt *statement) {
    if (statement == nullptr)
      return true;
    TraverseStmt(statement->getInit());
    TraverseStmt(statement->getConditionVariableDeclStmt());
    TraverseStmt(statement->getCond());
    ++conditionalDepth;
    TraverseStmt(statement->getBody());
    --conditionalDepth;
    return true;
  }

  bool TraverseLambdaExpr(clang::LambdaExpr *expression) {
    if (expression == nullptr)
      return true;
    ++conditionalDepth;
    TraverseStmt(expression->getBody());
    --conditionalDepth;
    return true;
  }

  bool TraverseConditionalOperator(
      clang::ConditionalOperator *expression) {
    if (expression == nullptr)
      return true;
    TraverseStmt(expression->getCond());
    ++conditionalDepth;
    TraverseStmt(expression->getTrueExpr());
    TraverseStmt(expression->getFalseExpr());
    --conditionalDepth;
    return true;
  }

  bool TraverseBinaryOperator(
      BinaryOperator *operation) {
    if (operation == nullptr)
      return true;
    TraverseStmt(operation->getLHS());
    if (operation->getOpcode() == clang::BO_LAnd ||
        operation->getOpcode() == clang::BO_LOr)
      ++conditionalDepth;
    TraverseStmt(operation->getRHS());
    if (operation->getOpcode() == clang::BO_LAnd ||
        operation->getOpcode() == clang::BO_LOr)
      --conditionalDepth;
    return true;
  }

  ASTContext *context = nullptr;
  unsigned conditionalDepth = 0;
  std::vector<FoundCall> calls;
};

bool semanticGlobalBinding(
    const FunctionDecl *function,
    const RouteBinding &binding,
    const std::set<const FunctionDecl *> &semanticKernels,
    const std::set<const EnumConstantDecl *> *choices,
    const SoftmaxKernelBindings *softmaxBindings) {
  function = function->getCanonicalDecl();
  if (semanticKernels.count(function) == 0)
    return false;
  const FunctionTemplateDecl *functionTemplate =
      primaryFunctionTemplate(function);
  if (functionTemplate == nullptr)
    return false;
  const TemplateParameterList *parameters =
      functionTemplate->getTemplateParameters();
  if (parameters == nullptr || parameters->size() < 3)
    return false;
  const auto *computeType =
      llvm::dyn_cast<TemplateTypeParmDecl>(
          parameters->getParam(2));
  if (computeType == nullptr ||
      binding.computeTypes.count(computeType) == 0)
    return false;
  if (softmaxBindings == nullptr)
    return true;
  const auto proven =
      softmaxBindings->find(function);
  if (proven == softmaxBindings->end())
    return false;
  unsigned matches = 0;
  for (const SoftmaxChoiceKey &evidence :
       proven->second) {
    const auto found =
        binding.enums.find(evidence.parameter);
    if (found != binding.enums.end() &&
        found->second == evidence.constant &&
        choices != nullptr &&
        choices->count(evidence.constant) != 0)
      ++matches;
  }
  return matches == 1;
}

bool traceExecutableRoutes(
    const FunctionDecl *function,
    const RouteBinding &binding,
    const std::set<const FunctionDecl *> &semanticKernels,
    const std::set<const EnumConstantDecl *> *choices,
    const SoftmaxKernelBindings *softmaxBindings,
    ASTContext &context,
    std::set<const FunctionDecl *> &active,
    bool &foundKernel,
    unsigned depth) {
  if (function == nullptr || depth > 48)
    return false;
  function = function->getCanonicalDecl();
  if (!active.insert(function).second)
    return true;
  ExecutableCallCollector collector;
  collector.context = &context;
  collector.TraverseStmt(
      const_cast<Stmt *>(function->getBody()));
  for (const ExecutableCallCollector::FoundCall &found :
       collector.calls) {
    std::set<const FunctionDecl *> candidates;
    if (!callCandidateFunctions(
            found.call, function, candidates) &&
        !unresolvedFunctorCandidates(
            found.call, candidates))
      continue;
    for (const FunctionDecl *candidate : candidates) {
      const RouteBinding child =
          bindRouteCall(found.call, candidate, binding);
      if (candidate->hasAttr<clang::CUDAGlobalAttr>()) {
        if (found.conditionalDepth != 0 ||
            !semanticGlobalBinding(
                candidate, child, semanticKernels, choices,
                softmaxBindings)) {
          active.erase(function);
          return false;
        }
        foundKernel = true;
        continue;
      }
      const FunctionDecl *definition = nullptr;
      if (!candidate->hasBody(definition) ||
          definition == nullptr)
        continue;
      bool childFound = false;
      if (!traceExecutableRoutes(
              definition, child, semanticKernels, choices,
              softmaxBindings,
              context, active, childFound, depth + 1)) {
        active.erase(function);
        return false;
      }
      if (!child.computeTypes.empty() && !childFound) {
        active.erase(function);
        return false;
      }
      foundKernel = foundKernel || childFound;
    }
  }
  active.erase(function);
  return true;
}

bool allForwardingRoutesReachSemanticKernels(
    const FunctionDecl *function,
    unsigned forwardedParameterCount,
    const std::set<const FunctionDecl *> &semanticKernels,
    const FunctionGraph &graph,
    const std::set<const EnumConstantDecl *> *choices,
    const SoftmaxKernelBindings *softmaxBindings,
    ASTContext &context) {
  (void)graph;
  ExecutableCallCollector collector;
  collector.context = &context;
  collector.TraverseStmt(
      const_cast<Stmt *>(function->getBody()));
  std::vector<const clang::CallExpr *> forwardingCalls;
  for (const ExecutableCallCollector::FoundCall &found :
       collector.calls) {
    if (callForwardsParameters(
            found.call, function,
            forwardedParameterCount))
      forwardingCalls.push_back(found.call);
  }
  if (forwardingCalls.empty())
    return false;
  for (const clang::CallExpr *call : forwardingCalls) {
    std::set<const FunctionDecl *> candidates;
    if (!callCandidateFunctions(
            call, function, candidates) &&
        !unresolvedFunctorCandidates(call, candidates))
      return false;
    for (const FunctionDecl *candidate : candidates) {
      RouteBinding rootBinding;
      const FunctionTemplateDecl *dispatcherTemplate =
          function->getDescribedFunctionTemplate();
      rootBinding.computeTypes.insert(
          llvm::cast<TemplateTypeParmDecl>(
              dispatcherTemplate->getTemplateParameters()
                  ->getParam(2)));
      const RouteBinding child =
          bindRouteCall(call, candidate, rootBinding);
      bool candidateFoundKernel = false;
      std::set<const FunctionDecl *> active;
      if (!traceExecutableRoutes(
              candidate, child, semanticKernels, choices,
              softmaxBindings,
              context, active, candidateFoundKernel, 0) ||
          !candidateFoundKernel)
        return false;
    }
  }
  return true;
}

bool standardCudaStatusType(QualType type,
                            SourceManager &sourceManager) {
  if (type.isNull())
    return false;
  type = type.getNonReferenceType();
  if (const auto *typedefType =
          type->getAs<clang::TypedefType>()) {
    const clang::TypedefNameDecl *declaration =
        typedefType->getDecl();
    if (declaration->getName() == "cudaError_t" &&
        sourceManager.isInSystemHeader(
            sourceManager.getExpansionLoc(
                declaration->getLocation())))
      return true;
  }
  const auto *enumType = type->getAs<clang::EnumType>();
  if (enumType == nullptr)
    return false;
  const clang::EnumDecl *declaration = enumType->getDecl();
  return declaration->getName() == "cudaError" &&
         sourceManager.isInSystemHeader(
             sourceManager.getExpansionLoc(
                 declaration->getLocation()));
}

bool dependentEnableIfStatus(
    QualType type,
    SourceManager &sourceManager) {
  const auto *dependent =
      llvm::dyn_cast_or_null<clang::DependentNameType>(
          type.getTypePtr());
  if (dependent == nullptr ||
      dependent->getIdentifier() == nullptr ||
      dependent->getIdentifier()->getName() != "type")
    return false;
  const clang::NestedNameSpecifier qualifier =
      dependent->getQualifier();
  if (!qualifier ||
      qualifier.getKind() !=
          clang::NestedNameSpecifier::Kind::Type)
    return false;
  const auto *specialization =
      qualifier.getAsType()->getAs<
          clang::TemplateSpecializationType>();
  if (specialization == nullptr ||
      specialization->template_arguments().size() != 2)
    return false;
  const clang::TemplateDecl *templateDeclaration =
      specialization->getTemplateName().getAsTemplateDecl();
  if (templateDeclaration == nullptr ||
      templateDeclaration->getName() != "enable_if" ||
      !sourceManager.isInSystemHeader(
          sourceManager.getExpansionLoc(
              templateDeclaration->getLocation())))
    return false;
  const clang::TemplateArgument &status =
      specialization->template_arguments()[1];
  return status.getKind() == clang::TemplateArgument::Type &&
         standardCudaStatusType(
             status.getAsType(), sourceManager);
}

bool statusReturnCompatible(
    const FunctionDecl *function,
    SourceManager &sourceManager) {
  if (function == nullptr || !function->hasBody())
    return false;
  for (const clang::AnnotateAttr *annotation :
       function->specific_attrs<clang::AnnotateAttr>()) {
    if (annotation->getAnnotation() ==
        "ascify.semantic.status")
      return true;
  }
  const QualType type =
      function->getReturnType().getNonReferenceType();
  if (standardCudaStatusType(type, sourceManager))
    return true;
  return dependentEnableIfStatus(type, sourceManager);
}

bool isSoftmaxDispatcher(
    const FunctionDecl *function,
    ASTContext &context,
    SourceManager &sourceManager,
    const std::set<const EnumConstantDecl *> &choices,
    const std::set<const FunctionDecl *> &semanticKernels,
    const SoftmaxKernelBindings &softmaxBindings,
    const FunctionGraph &graph) {
  if (function == nullptr || !function->hasBody() ||
      choices.empty() ||
      function->hasAttr<clang::CUDADeviceAttr>() ||
      function->hasAttr<clang::CUDAGlobalAttr>() ||
      !statusReturnCompatible(function, sourceManager) ||
      function->getNumParams() != 5 ||
      !exactlyTypeTemplateParameters(function, 3) ||
      !function->getParamDecl(0)->getType()->isPointerType() ||
      !templateTypeParameter(
          function->getParamDecl(1)->getType(),
          0, 0, false, false, false) ||
      !templateTypeParameter(
          function->getParamDecl(2)->getType(),
          0, 1, false, false, false) ||
      !isIntegralWidthAtLeast(
          function->getParamDecl(3)->getType(), context, 32) ||
      !isIntegralWidthAtLeast(
          function->getParamDecl(4)->getType(), context, 32))
    return false;
  if (!hasRowWidthRoutingShape(function, context, 4, 5))
    return false;

  return allForwardingRoutesReachSemanticKernels(
      function, 5, semanticKernels, graph, &choices,
      &softmaxBindings,
      context);
}

const Decl *directReferenceThroughCasts(const Expr *expression) {
  expression = stripExpr(expression);
  while (const auto *cast =
             llvm::dyn_cast_or_null<clang::ExplicitCastExpr>(
                 expression))
    expression = stripExpr(cast->getSubExpr());
  return directReferencedDecl(expression);
}

struct SquareEvidence {
  const Decl *sumRoot = nullptr;
  const Decl *valueRoot = nullptr;
  const Expr *value = nullptr;
  const Stmt *operation = nullptr;
  SourceLocation location;
};

class SquareCollector
    : public clang::RecursiveASTVisitor<SquareCollector> {
public:
  explicit SquareCollector(ASTContext &context)
      : context(context) {}

  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (discardedOrDeferredExecution(
            operation, context))
      return true;
    if (operation->getOpcode() != clang::BO_AddAssign)
      return true;
    const auto *multiply =
        llvm::dyn_cast_or_null<BinaryOperator>(
            stripExpr(operation->getRHS()));
    record(
        operation->getLHS(), multiply,
        nullptr, operation,
        operation->getExprLoc());
    return true;
  }

  bool VisitVarDecl(VarDecl *variable) {
    if (!variable->hasInit())
      return true;
    if (discardedOrDeferredExecution(
            variable->getInit(), context))
      return true;
    const auto *multiply =
        llvm::dyn_cast_or_null<BinaryOperator>(
            stripExpr(variable->getInit()));
    record(
        nullptr, multiply, variable,
        variable->getInit(),
        variable->getLocation());
    return true;
  }

  void record(const Expr *sumExpression,
              const BinaryOperator *multiply,
              const VarDecl *sumVariable,
              const Stmt *operation,
              SourceLocation location) {
    if (multiply == nullptr ||
        multiply->getOpcode() != clang::BO_Mul ||
        !sameReferenceExpression(
            multiply->getLHS(), multiply->getRHS()))
      return;
    const Decl *sumRoot =
        sumVariable != nullptr
            ? static_cast<const Decl *>(sumVariable)
            : storageRoot(sumExpression);
    const Decl *valueRoot =
        storageRoot(multiply->getLHS());
    if (sumRoot != nullptr && valueRoot != nullptr)
      squares.push_back(
          {sumRoot, valueRoot,
           multiply->getLHS(), operation,
           location});
  }

  ASTContext &context;
  std::vector<SquareEvidence> squares;
};

struct MeanEvidence {
  ProducedValue produced;
  const Decl *numeratorRoot = nullptr;
  const Stmt *operation = nullptr;
};

std::vector<MeanEvidence> collectMeans(
    const Stmt *body,
    const ParmVarDecl *columns,
    PrimitiveRegistry &registry,
    ASTContext &context) {
  std::vector<MeanEvidence> means;
  auto record =
      [&](const Expr *value,
          const Expr *numerator,
          const Expr *denominator) {
        if (directReferenceThroughCasts(denominator) != columns)
          return;
        const Decl *numeratorRoot = storageRoot(numerator);
        if (numeratorRoot == nullptr)
          return;
        ProducerFinder producer(value);
        producer.TraverseStmt(const_cast<Stmt *>(body));
        if (producer.produced.matches == 1 &&
            producedOnExecutablePath(
                producer.produced, context))
          means.push_back(
              {producer.produced, numeratorRoot,
               value});
      };

  PrimitiveCallCollector calls(registry, context);
  calls.TraverseStmt(const_cast<Stmt *>(body));
  for (const clang::CallExpr *call : calls.divideCalls) {
    if (call->getNumArgs() == 2)
      record(call, call->getArg(0), call->getArg(1));
  }
  DivisionCollector divisions;
  divisions.TraverseStmt(const_cast<Stmt *>(body));
  for (const BinaryOperator *division : divisions.divisions) {
    if (discardedOrDeferredExecution(
            division, context))
      continue;
    record(
        division, division->getLHS(), division->getRHS());
  }
  return means;
}

bool meanPlusEpsilon(const Expr *expression,
                     const MeanEvidence &mean,
                     const ParmVarDecl *epsilon) {
  const auto *addition =
      llvm::dyn_cast_or_null<BinaryOperator>(
          stripExpr(expression));
  if (addition == nullptr ||
      addition->getOpcode() != clang::BO_Add)
    return false;
  const bool lhsMean =
      referencesProducedValue(addition->getLHS(), mean.produced);
  const bool rhsMean =
      referencesProducedValue(addition->getRHS(), mean.produced);
  const bool lhsEpsilon =
      directReferenceThroughCasts(addition->getLHS()) == epsilon;
  const bool rhsEpsilon =
      directReferenceThroughCasts(addition->getRHS()) == epsilon;
  return (lhsMean && rhsEpsilon) ||
         (rhsMean && lhsEpsilon);
}

class InverseUseVisitor
    : public clang::RecursiveASTVisitor<InverseUseVisitor> {
public:
  struct Normalization {
    const Decl *sourceRoot = nullptr;
    const Decl *outputRoot = nullptr;
    const Expr *sourceValue = nullptr;
    const BinaryOperator *operation = nullptr;
  };

  InverseUseVisitor(
      const ProducedValue &inverse,
      const ParmVarDecl *inverseOutput,
      ASTContext &context)
      : inverse(inverse), inverseOutput(inverseOutput),
        context(context) {}

  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (discardedOrDeferredExecution(
            operation, context))
      return true;
    if (operation->getOpcode() == clang::BO_MulAssign &&
        referencesProducedValue(
            operation->getRHS(), inverse)) {
      const Decl *root =
          storageRoot(operation->getLHS());
      if (root != nullptr)
        normalizations.push_back(
            {root, root, operation->getLHS(),
             operation});
    }
    if (operation->getOpcode() == clang::BO_Assign) {
      if (storageRoot(operation->getLHS()) ==
              inverseOutput &&
          referencesProducedValue(
              operation->getRHS(), inverse)) {
        writesInverse = true;
        inverseWrites.push_back(operation);
      }
      const auto *multiply =
          llvm::dyn_cast_or_null<BinaryOperator>(
              stripExpr(operation->getRHS()));
      if (multiply != nullptr &&
          multiply->getOpcode() == clang::BO_Mul) {
        const Expr *operands[2] = {
            multiply->getLHS(), multiply->getRHS()};
        for (unsigned inverseIndex = 0;
             inverseIndex != 2; ++inverseIndex) {
          if (!referencesProducedValue(
                  operands[inverseIndex], inverse))
            continue;
          const Decl *source =
              directStorageReadRoot(
                  operands[1 - inverseIndex]);
          const Decl *output =
              storageRoot(operation->getLHS());
          if (source != nullptr && output != nullptr)
            normalizations.push_back(
                {source, output,
                 operands[1 - inverseIndex],
                 operation});
        }
      }
    }
    return true;
  }

  const ProducedValue &inverse;
  const ParmVarDecl *inverseOutput;
  ASTContext &context;
  bool writesInverse = false;
  std::vector<const BinaryOperator *> inverseWrites;
  std::vector<Normalization> normalizations;
};

class MutationBetweenVisitor
    : public clang::RecursiveASTVisitor<MutationBetweenVisitor> {
public:
  MutationBetweenVisitor(
      const Decl *root,
      const BinaryOperator *allowed,
      SourceLocation begin,
      SourceLocation end,
      SourceManager &sourceManager,
      const SoftmaxChoiceKey *choice,
      const std::set<const clang::CallExpr *>
          *allowedReadCalls = nullptr)
      : root(root), allowed(allowed), begin(begin), end(end),
        sourceManager(sourceManager), choice(choice),
        allowedReadCalls(allowedReadCalls) {}

  bool TraverseIfStmt(IfStmt *statement) {
    if (statement == nullptr)
      return true;
    TraverseStmt(statement->getInit());
    TraverseStmt(statement->getConditionVariableDeclStmt());
    TraverseStmt(statement->getCond());
    if (choice != nullptr) {
      const NonTypeTemplateParmDecl *parameter = nullptr;
      const EnumConstantDecl *constant = nullptr;
      if (enumEquality(
              statement->getCond(), parameter, constant) &&
          parameter == choice->parameter) {
        return TraverseStmt(
            constant == choice->constant
                ? statement->getThen()
                : statement->getElse());
      }
    }
    TraverseStmt(statement->getThen());
    TraverseStmt(statement->getElse());
    return true;
  }

  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (!operation->isAssignmentOp() ||
        operation == allowed ||
        storageRoot(operation->getLHS()) != root)
      return true;
    const SourceLocation location =
        sourceManager.getExpansionLoc(operation->getExprLoc());
    if (sourceManager.isBeforeInTranslationUnit(begin, location) &&
        sourceManager.isBeforeInTranslationUnit(location, end))
      found = true;
    return !found;
  }

  bool VisitUnaryOperator(clang::UnaryOperator *operation) {
    if (operation->getOpcode() != clang::UO_PreInc &&
        operation->getOpcode() != clang::UO_PostInc &&
        operation->getOpcode() != clang::UO_PreDec &&
        operation->getOpcode() != clang::UO_PostDec)
      return true;
    if (storageRoot(operation->getSubExpr()) == root)
      record(operation->getExprLoc());
    return !found;
  }

  bool VisitCallExpr(clang::CallExpr *call) {
    if (allowedReadCalls != nullptr &&
        allowedReadCalls->count(call) != 0)
      return true;
    for (const Expr *argument : call->arguments()) {
      if (storageRoot(argument) == root ||
          containsDecl(argument, root)) {
        record(call->getExprLoc());
        break;
      }
    }
    return !found;
  }

  void record(SourceLocation rawLocation) {
    const SourceLocation location =
        sourceManager.getExpansionLoc(rawLocation);
    if (sourceManager.isBeforeInTranslationUnit(begin, location) &&
        sourceManager.isBeforeInTranslationUnit(location, end))
      found = true;
  }

  const Decl *root;
  const BinaryOperator *allowed;
  SourceLocation begin;
  SourceLocation end;
  SourceManager &sourceManager;
  const SoftmaxChoiceKey *choice;
  const std::set<const clang::CallExpr *>
      *allowedReadCalls;
  bool found = false;
};

bool normalizationReachesStore(
    const Stmt *body,
    const Decl *root,
    const BinaryOperator *normalization,
    const std::vector<AdapterBufferEvidence::Access> &stores,
    SourceManager &sourceManager,
    const SoftmaxChoiceKey *choice) {
  if (root == nullptr || normalization == nullptr)
    return false;
  const SourceLocation begin =
      sourceManager.getExpansionLoc(
          normalization->getEndLoc());
  SourceLocation firstStore;
  for (const AdapterBufferEvidence::Access &store : stores) {
    if (store.root != root || store.call == nullptr)
      continue;
    const SourceLocation location =
        sourceManager.getExpansionLoc(
            store.call->getBeginLoc());
    if (!sourceManager.isBeforeInTranslationUnit(begin, location))
      continue;
    if (firstStore.isInvalid() ||
        sourceManager.isBeforeInTranslationUnit(
            location, firstStore))
      firstStore = location;
  }
  if (firstStore.isInvalid())
    return false;
  MutationBetweenVisitor mutations(
      root, normalization, begin,
      firstStore, sourceManager, choice);
  mutations.TraverseStmt(const_cast<Stmt *>(body));
  return !mutations.found;
}

bool normalizationReachesSpecificStore(
    const Stmt *body,
    const Decl *root,
    const BinaryOperator *normalization,
    const AdapterBufferEvidence::Access &store,
    SourceManager &sourceManager,
    const SoftmaxChoiceKey *choice,
    const std::set<const clang::CallExpr *>
        *allowedReadCalls) {
  if (body == nullptr || root == nullptr ||
      normalization == nullptr ||
      store.root != root || store.call == nullptr)
    return false;
  const SourceLocation begin =
      sourceManager.getExpansionLoc(
          normalization->getEndLoc());
  const SourceLocation end =
      sourceManager.getExpansionLoc(
          store.call->getBeginLoc());
  if (!sourceManager.isBeforeInTranslationUnit(
          begin, end))
    return false;
  MutationBetweenVisitor mutations(
      root, normalization, begin, end,
      sourceManager, choice, allowedReadCalls);
  mutations.TraverseStmt(const_cast<Stmt *>(body));
  return !mutations.found;
}

bool cudaThreadLaneZeroCondition(
    const Expr *condition,
    ASTContext &context) {
  const auto *comparison =
      llvm::dyn_cast_or_null<BinaryOperator>(
          stripExpr(condition));
  if (comparison == nullptr ||
      comparison->getOpcode() != clang::BO_EQ)
    return false;
  const Expr *lane = nullptr;
  if (evaluatesToZero(
          comparison->getLHS(), context))
    lane = stripExpr(comparison->getRHS());
  else if (evaluatesToZero(
               comparison->getRHS(), context))
    lane = stripExpr(comparison->getLHS());
  const clang::MemberExpr *member =
      llvm::dyn_cast_or_null<clang::MemberExpr>(lane);
  if (const auto *pseudo =
          llvm::dyn_cast_or_null<
              clang::PseudoObjectExpr>(lane)) {
    const auto *fetch =
        llvm::dyn_cast_or_null<clang::CallExpr>(
            stripExpr(pseudo->getResultExpr()));
    member =
        llvm::dyn_cast_or_null<clang::MemberExpr>(
            stripExpr(
                fetch == nullptr
                    ? nullptr
                    : fetch->getCallee()));
  }
  if (member == nullptr ||
      (member->getMemberNameInfo().getAsString() != "x" &&
       member->getMemberNameInfo().getAsString() !=
           "__fetch_builtin_x"))
    return false;
  ReferencedVariableSet references;
  references.TraverseStmt(
      const_cast<Expr *>(lane));
  if (references.variables.size() != 1)
    return false;
  const VarDecl *builtin =
      *references.variables.begin();
  const CXXRecordDecl *record =
      builtin->getType()
          .getNonReferenceType()
          .getUnqualifiedType()
          ->getAsCXXRecordDecl();
  const SourceManager &sourceManager =
      context.getSourceManager();
  return builtin->getName() == "threadIdx" &&
         cudaExecutionBuiltin(builtin) &&
         record != nullptr &&
         sourceManager.isInSystemHeader(
             builtin->getLocation()) &&
         sourceManager.isInSystemHeader(
             record->getLocation()) &&
         sourceManager.isInSystemHeader(
             member->getMemberDecl()->getLocation());
}

bool singleThreadLeaderWrite(
    const BinaryOperator *write,
    ASTContext &context) {
  if (write == nullptr)
    return false;
  const Stmt *current = write;
  const IfStmt *leader = nullptr;
  std::set<const Stmt *> active;
  while (current != nullptr &&
         active.insert(current).second) {
    const auto parents = context.getParents(*current);
    const Stmt *next = nullptr;
    for (const clang::DynTypedNode &parent : parents) {
      if (const auto *conditional =
              parent.get<IfStmt>()) {
        if (conditional->getThen() == current ||
            conditional->getElse() == current) {
          if (leader != nullptr)
            return false;
          leader = conditional;
        }
      }
      if (next == nullptr)
        next = parent.get<Stmt>();
    }
    current = next;
  }
  return leader != nullptr &&
         leader->getElse() == nullptr &&
         stripAttributedStmt(
             onlyStatement(leader->getThen())) == write &&
         cudaThreadLaneZeroCondition(
             leader->getCond(), context);
}

bool rmsPipelineExecutable(
    const Stmt *body,
    const SquareEvidence &square,
    const MeanEvidence &mean,
    const clang::CallExpr *rsqrt,
    const ProducedValue &inverse,
    const std::vector<const BinaryOperator *> &inverseWrites,
    const InverseUseVisitor::Normalization &normalization,
    const AdapterBufferEvidence &adapters,
    const std::set<const clang::CallExpr *>
        &reductionCalls,
    ASTContext &context,
    SourceManager &sourceManager) {
  for (const AdapterBufferEvidence::Access &squareLoad :
       adapters.loadCalls) {
    if (!adapterAccessReachesBefore(
            body, squareLoad, square.valueRoot,
            square.operation, sourceManager))
      continue;
    for (const AdapterBufferEvidence::Access
             &normalizationLoad : adapters.loadCalls) {
      if (!adapterAccessReachesBefore(
              body, normalizationLoad,
              normalization.sourceRoot,
              normalization.operation,
              sourceManager))
        continue;
      for (const AdapterBufferEvidence::Access &store :
           adapters.storeCalls) {
        if (!normalizationReachesSpecificStore(
                body, normalization.outputRoot,
                normalization.operation, store,
                sourceManager, nullptr))
          continue;
        for (const BinaryOperator *inverseWrite :
             inverseWrites) {
          if (!singleThreadLeaderWrite(
                  inverseWrite, context))
            continue;
          const std::vector<const Stmt *> operations = {
              squareLoad.call, square.operation,
              mean.operation, rsqrt,
              producedStatement(inverse),
              normalizationLoad.call,
              normalization.operation, store.call};
          std::vector<const Stmt *> complete =
              operations;
          for (const clang::CallExpr *reduction :
               reductionCalls)
            complete.push_back(reduction);
          if (controlContextsCompatible(
                  complete, context))
            return true;
        }
      }
    }
  }
  return false;
}

bool rmsAdapterAccessTopologyExact(
    const Stmt *body,
    const std::vector<const SquareEvidence *> &squares,
    const InverseUseVisitor::Normalization &normalization,
    const AdapterBufferEvidence &adapters,
    SourceManager &sourceManager) {
  const bool exactCounts =
      (adapters.loadCalls.size() == 1 &&
       adapters.storeCalls.size() == 1) ||
      (adapters.loadCalls.size() == 2 &&
       (adapters.storeCalls.size() == 1 ||
        adapters.storeCalls.size() == 2));
  if (!exactCounts)
    return false;
  for (const AdapterBufferEvidence::Access &load :
       adapters.loadCalls) {
    bool relevant = adapterAccessReachesBefore(
        body, load, normalization.sourceRoot,
        normalization.operation, sourceManager);
    for (const SquareEvidence *square : squares) {
      relevant =
          relevant ||
          adapterAccessReachesBefore(
              body, load, square->valueRoot,
              square->operation, sourceManager);
    }
    if (!relevant)
      return false;
  }
  std::set<const clang::CallExpr *>
      canonicalStoreReads;
  for (const AdapterBufferEvidence::Access &store :
       adapters.storeCalls)
    canonicalStoreReads.insert(store.call);
  for (const AdapterBufferEvidence::Access &store :
       adapters.storeCalls) {
    if (!normalizationReachesSpecificStore(
            body, normalization.outputRoot,
            normalization.operation, store,
            sourceManager, nullptr,
            &canonicalStoreReads))
      return false;
  }
  return true;
}

bool proveRmsAlgebra(const FunctionDecl *function,
                     const Stmt *body,
                     const ParmVarDecl *columns,
                     const ParmVarDecl *epsilon,
                     const ParmVarDecl *inverseOutput,
                     PrimitiveRegistry &registry,
                     const AdapterBufferEvidence &adapters,
                     const std::set<const clang::CallExpr *>
                         &allowedStructuralCalls,
                     ASTContext &context,
                     SourceManager &sourceManager) {
  std::set<const clang::CallExpr *> allowedAdapterCalls;
  for (const AdapterBufferEvidence::Access &access :
       adapters.loadCalls)
    allowedAdapterCalls.insert(access.call);
  for (const AdapterBufferEvidence::Access &access :
       adapters.storeCalls)
    allowedAdapterCalls.insert(access.call);
  SquareCollector squareCollector(context);
  squareCollector.TraverseStmt(const_cast<Stmt *>(body));
  if (squareCollector.squares.empty())
    return false;
  const std::vector<MeanEvidence> means =
      collectMeans(body, columns, registry, context);
  if (means.empty())
    return false;

  PrimitiveCallCollector calls(registry, context);
  calls.TraverseStmt(const_cast<Stmt *>(body));
  for (const clang::CallExpr *call : calls.rsqrtCalls) {
    if (call->getNumArgs() != 1)
      continue;
    ProducerFinder inverseProducer(call);
    inverseProducer.TraverseStmt(const_cast<Stmt *>(body));
    if (inverseProducer.produced.matches != 1 ||
        !producedOnExecutablePath(
            inverseProducer.produced, context))
      continue;
    for (const MeanEvidence &mean : means) {
      if (!meanPlusEpsilon(
              call->getArg(0), mean, epsilon))
        continue;
      std::set<const clang::CallExpr *>
          allowedSemanticCalls =
              allowedStructuralCalls;
      std::vector<const SquareEvidence *> valueSquares;
      for (const SquareEvidence &square :
           squareCollector.squares) {
        const bool squareCompute =
            expressionHasTemplateElementType(
                square.value, 0, 2);
        const bool squareReduction =
            storageFlowsThroughCall(
                body, square.sumRoot,
                mean.numeratorRoot,
                mean.operation,
                function, context,
                &allowedSemanticCalls);
        const bool squareLoaded =
            adapters.loaded.count(
                square.valueRoot) != 0;
        const bool squareOrdered =
            hasAdapterAccessBeforeLocation(
                adapters.loadCalls,
                square.valueRoot,
                square.location, sourceManager);
        if (squareCompute && squareReduction &&
            squareLoaded && squareOrdered)
          valueSquares.push_back(&square);
      }
      if (valueSquares.empty())
        continue;
      InverseUseVisitor uses(
          inverseProducer.produced,
          inverseOutput, context);
      uses.TraverseStmt(const_cast<Stmt *>(body));
      if (uses.inverseWrites.size() != 1)
        continue;
      unsigned validNormalizations = 0;
      for (const InverseUseVisitor::Normalization
               &normalization : uses.normalizations) {
        const bool computeSource =
            expressionHasTemplateElementType(
                normalization.sourceValue, 0, 2);
        const bool normalizationLoaded =
            adapterLoadReachesBefore(
                body, adapters,
                normalization.sourceRoot,
                normalization.operation,
                sourceManager);
        if (!computeSource || !normalizationLoaded)
          continue;
        bool relatedToSquares = false;
        bool executablePipeline = false;
        for (const SquareEvidence *square :
             valueSquares) {
          const Decl *valueRoot =
              square->valueRoot;
          if (valueRoot ==
                  normalization.sourceRoot ||
              storageCopiesReachBefore(
                  body, valueRoot,
                  normalization.sourceRoot,
                  normalization.operation,
                  sourceManager) ||
              equivalentAdapterLoads(
                  adapters, valueRoot,
                  normalization.sourceRoot,
                  context)) {
            relatedToSquares = true;
            if (rmsPipelineExecutable(
                    body, *square, mean, call,
                    inverseProducer.produced,
                    uses.inverseWrites, normalization,
                    adapters, allowedSemanticCalls,
                    context,
                    sourceManager))
              executablePipeline = true;
          }
        }
        const bool stored =
            adapters.stored.count(
                normalization.outputRoot) != 0;
        const bool reachesStore =
            stored &&
            normalizationReachesStore(
                body, normalization.outputRoot,
                normalization.operation,
                adapters.storeCalls,
                sourceManager, nullptr);
        const bool exactAccessTopology =
            rmsAdapterAccessTopologyExact(
                body, valueSquares, normalization,
                adapters, sourceManager);
        if (!relatedToSquares || !stored ||
            !reachesStore || !executablePipeline ||
            !exactAccessTopology ||
            !semanticEffectsAllowed(
                function, registry,
                allowedAdapterCalls,
                allowedSemanticCalls,
                inverseOutput,
                uses.inverseWrites.front(),
                nullptr, context))
          continue;
        ++validNormalizations;
      }
      if (uses.writesInverse &&
          validNormalizations == 1)
        return true;
    }
  }
  return false;
}

bool isRmsKernel(const FunctionDecl *function,
                 ASTContext &context,
                 PrimitiveRegistry &registry,
                 SourceManager &sourceManager) {
  if (function == nullptr || !function->hasBody() ||
      !function->hasAttr<clang::CUDAGlobalAttr>() ||
      function->getNumParams() != 6 ||
      !templateTypeParameter(
          function->getParamDecl(0)->getType(),
          0, 0, false, false, false) ||
      !templateTypeParameter(
          function->getParamDecl(1)->getType(),
          0, 1, false, false, false) ||
      !isIntegralWidthAtLeast(
          function->getParamDecl(2)->getType(), context, 32) ||
      !isIntegralWidthAtLeast(
          function->getParamDecl(3)->getType(), context, 32) ||
      !function->getParamDecl(4)->getType()
           .getNonReferenceType()->isFloatingType() ||
      !templateTypeParameter(
          function->getParamDecl(5)->getType(),
          0, 2, true, false, true))
    return false;

  AdapterCallVisitor adapterCalls(
      function->getParamDecl(0), function->getParamDecl(1),
      context);
  adapterCalls.TraverseStmt(
      const_cast<Stmt *>(function->getBody()));
  if (!adapterCalls.proven())
    return false;
  AdapterBufferVisitor adapterBuffers(
      function->getParamDecl(0),
      function->getParamDecl(1), context);
  adapterBuffers.TraverseStmt(
      const_cast<Stmt *>(function->getBody()));
  std::set<const clang::CallExpr *>
      allowedStructuralCalls;
  if (!canonicalKernelCoverageAndControl(
          function, false,
          adapterBuffers.evidence, context,
          allowedStructuralCalls))
    return false;
  return proveRmsAlgebra(
      function, function->getBody(),
      function->getParamDecl(3),
      function->getParamDecl(4),
      function->getParamDecl(5),
      registry, adapterBuffers.evidence,
      allowedStructuralCalls,
      context, sourceManager);
}

enum class DirectKernelDomain {
  None,
  Block,
  SoftmaxWarp,
  RmsNormWarp,
};

struct DirectLaunchWrapperProof {
  bool softmax = false;
  const FunctionDecl *function = nullptr;
  const ParmVarDecl *stream = nullptr;
  const ParmVarDecl *load = nullptr;
  const ParmVarDecl *store = nullptr;
  const ParmVarDecl *rows = nullptr;
  const ParmVarDecl *columns = nullptr;
  const ParmVarDecl *epsilon = nullptr;
  const ParmVarDecl *inverseOutput = nullptr;
  const ParmVarDecl *sharedMemory = nullptr;
  const TemplateTypeParmDecl *computeType = nullptr;
  const NonTypeTemplateParmDecl *packSize = nullptr;
  const NonTypeTemplateParmDecl *maxColumnsPerThread =
      nullptr;
  const NonTypeTemplateParmDecl *minColumnsPerThread =
      nullptr;
  const NonTypeTemplateParmDecl *threadGroupWidth =
      nullptr;
  const NonTypeTemplateParmDecl *rowsPerAccess = nullptr;
  const NonTypeTemplateParmDecl *padding = nullptr;
  const NonTypeTemplateParmDecl *algorithm = nullptr;
  const EnumConstantDecl *choice = nullptr;
  const clang::CompoundStmt *postGeometryBlock = nullptr;
  DirectKernelDomain domain = DirectKernelDomain::None;
  bool rowsIntegralCast = false;
  bool columnsIntegralCast = false;

  bool proven() const {
    return function != nullptr && stream != nullptr &&
           load != nullptr && store != nullptr &&
           rows != nullptr && columns != nullptr &&
           computeType != nullptr &&
           packSize != nullptr &&
           domain != DirectKernelDomain::None &&
           (softmax ||
            (epsilon != nullptr &&
             inverseOutput != nullptr &&
             postGeometryBlock != nullptr));
  }
};

const ParmVarDecl *parameterThroughIntegralCasts(
    const Expr *expression,
    bool &sawIntegralCast) {
  sawIntegralCast = false;
  expression =
      expression == nullptr
          ? nullptr
          : expression->IgnoreParenImpCasts();
  while (const auto *cast =
             llvm::dyn_cast_or_null<
                 clang::ExplicitCastExpr>(expression)) {
    if (cast->getCastKind() != clang::CK_IntegralCast &&
        cast->getCastKind() != clang::CK_NoOp)
      return nullptr;
    sawIntegralCast =
        sawIntegralCast ||
        cast->getCastKind() == clang::CK_IntegralCast;
    expression =
        cast->getSubExpr()->IgnoreParenImpCasts();
  }
  return directReferencedAs<ParmVarDecl>(expression);
}

bool typeIsTemplateParameter(
    QualType type,
    const TemplateTypeParmDecl *parameter,
    bool pointer) {
  if (type.isNull() || parameter == nullptr)
    return false;
  type = type.getNonReferenceType();
  if (pointer) {
    const auto *pointerType =
        type->getAs<clang::PointerType>();
    if (pointerType == nullptr)
      return false;
    type = pointerType->getPointeeType();
  }
  type = type.getUnqualifiedType();
  const auto *templateType =
      type->getAs<clang::TemplateTypeParmType>();
  return templateType != nullptr &&
         templateType->getDecl() == parameter;
}

bool wrapperTemplateMapping(
    const FunctionDecl *wrapper,
    const clang::CUDAKernelCallExpr *launch,
    const FunctionDecl *kernel,
    const TemplateTypeParmDecl *&loadType,
    const TemplateTypeParmDecl *&storeType,
    const TemplateTypeParmDecl *&computeType,
    std::vector<clang::TemplateArgument> &arguments) {
  const FunctionTemplateDecl *wrapperTemplate =
      wrapper == nullptr
          ? nullptr
          : wrapper->getDescribedFunctionTemplate();
  const FunctionTemplateDecl *kernelTemplate =
      primaryFunctionTemplate(kernel);
  if (wrapperTemplate == nullptr ||
      kernelTemplate == nullptr)
    return false;
  const TemplateParameterList *wrapperParameters =
      wrapperTemplate->getTemplateParameters();
  const TemplateParameterList *kernelParameters =
      kernelTemplate->getTemplateParameters();
  if (wrapperParameters == nullptr ||
      wrapperParameters->size() < 3 ||
      kernelParameters == nullptr ||
      kernelParameters->size() < 3)
    return false;
  loadType = llvm::dyn_cast<TemplateTypeParmDecl>(
      wrapperParameters->getParam(0));
  storeType = llvm::dyn_cast<TemplateTypeParmDecl>(
      wrapperParameters->getParam(1));
  computeType = llvm::dyn_cast<TemplateTypeParmDecl>(
      wrapperParameters->getParam(2));
  if (loadType == nullptr || storeType == nullptr ||
      computeType == nullptr ||
      loadType->getIdentifier() == nullptr ||
      storeType->getIdentifier() == nullptr ||
      computeType->getIdentifier() == nullptr ||
      loadType->isParameterPack() ||
      storeType->isParameterPack() ||
      computeType->isParameterPack())
    return false;
  arguments = callTemplateArguments(launch, kernel);
  if (arguments.size() < 3)
    return false;
  const TemplateTypeParmDecl *mapped[3] = {};
  for (unsigned index = 0; index != 3; ++index) {
    if (arguments[index].getKind() !=
        clang::TemplateArgument::Type)
      return false;
    mapped[index] =
        referencedTypeParameter(
            arguments[index].getAsType());
  }
  return mapped[0] == loadType &&
         mapped[1] == storeType &&
         mapped[2] == computeType;
}

const NonTypeTemplateParmDecl *directWrapperNonTypeArgument(
    const clang::TemplateArgument &argument,
    const FunctionDecl *wrapper,
    bool boolean) {
  if (argument.getKind() !=
      clang::TemplateArgument::Expression)
    return nullptr;
  const auto *parameter =
      directReferencedAs<NonTypeTemplateParmDecl>(
          argument.getAsExpr());
  const FunctionTemplateDecl *wrapperTemplate =
      wrapper == nullptr
          ? nullptr
          : wrapper->getDescribedFunctionTemplate();
  const TemplateParameterList *parameters =
      wrapperTemplate == nullptr
          ? nullptr
          : wrapperTemplate->getTemplateParameters();
  if (parameter == nullptr || parameters == nullptr ||
      parameter->getIdentifier() == nullptr ||
      parameter->isParameterPack() ||
      (boolean
           ? !parameter->getType()
                  .getNonReferenceType()
                  .getUnqualifiedType()
                  ->isBooleanType()
           : !parameter->getType()->isIntegerType()))
    return nullptr;
  for (const NamedDecl *candidate : *parameters) {
    if (candidate == parameter)
      return parameter;
  }
  return nullptr;
}

bool directKernelDomainMapping(
    const FunctionDecl *wrapper,
    const FunctionDecl *kernel,
    const std::vector<clang::TemplateArgument> &arguments,
    bool softmax,
    DirectLaunchWrapperProof &proof) {
  const FunctionTemplateDecl *kernelTemplate =
      primaryFunctionTemplate(kernel);
  const TemplateParameterList *parameters =
      kernelTemplate == nullptr
          ? nullptr
          : kernelTemplate->getTemplateParameters();
  if (parameters == nullptr ||
      arguments.size() != parameters->size())
    return false;
  proof.packSize = directWrapperNonTypeArgument(
      arguments[3], wrapper, false);
  if (proof.packSize == nullptr)
    return false;
  const unsigned blockArity = softmax ? 6 : 5;
  if (parameters->size() == blockArity) {
    proof.domain = DirectKernelDomain::Block;
    return true;
  }
  if (parameters->size() != 9)
    return false;
  proof.maxColumnsPerThread =
      directWrapperNonTypeArgument(
          arguments[4], wrapper, false);
  proof.threadGroupWidth =
      directWrapperNonTypeArgument(
          arguments[softmax ? 5 : 6],
          wrapper, false);
  proof.rowsPerAccess =
      directWrapperNonTypeArgument(
          arguments[softmax ? 6 : 7],
          wrapper, false);
  proof.padding =
      directWrapperNonTypeArgument(
          arguments[softmax ? 7 : 8],
          wrapper, true);
  if (!softmax)
    proof.minColumnsPerThread =
        directWrapperNonTypeArgument(
            arguments[5], wrapper, false);
  if (proof.maxColumnsPerThread == nullptr ||
      proof.threadGroupWidth == nullptr ||
      proof.rowsPerAccess == nullptr ||
      proof.padding == nullptr ||
      (!softmax &&
       proof.minColumnsPerThread == nullptr))
    return false;
  proof.domain =
      softmax
          ? DirectKernelDomain::SoftmaxWarp
          : DirectKernelDomain::RmsNormWarp;
  return true;
}

bool softmaxLaunchChoice(
    const FunctionDecl *kernel,
    const std::vector<clang::TemplateArgument> &arguments,
    const SoftmaxKernelBindings &bindings,
    const NonTypeTemplateParmDecl *&wrapperAlgorithm,
    const EnumConstantDecl *&choice) {
  wrapperAlgorithm = nullptr;
  choice = nullptr;
  const auto proven =
      bindings.find(kernel->getCanonicalDecl());
  const FunctionTemplateDecl *kernelTemplate =
      primaryFunctionTemplate(kernel);
  if (proven == bindings.end() ||
      kernelTemplate == nullptr)
    return false;
  const TemplateParameterList *parameters =
      kernelTemplate->getTemplateParameters();
  unsigned matches = 0;
  for (const SoftmaxChoiceKey &evidence :
       proven->second) {
    unsigned index = 0;
    while (index < parameters->size() &&
           parameters->getParam(index) !=
               evidence.parameter)
      ++index;
    if (index >= parameters->size() ||
        index >= arguments.size())
      continue;
    const clang::TemplateArgument &argument =
        arguments[index];
    const Expr *expression =
        argument.getKind() ==
                clang::TemplateArgument::Expression
            ? stripExpr(argument.getAsExpr())
            : nullptr;
    const EnumConstantDecl *constant =
        argument.getKind() ==
                clang::TemplateArgument::Declaration
            ? llvm::dyn_cast<EnumConstantDecl>(
                  argument.getAsDecl())
            : directReferencedAs<EnumConstantDecl>(
                  expression);
    const auto *algorithm =
        directReferencedAs<
            NonTypeTemplateParmDecl>(expression);
    if (constant == evidence.constant) {
      ++matches;
      wrapperAlgorithm = nullptr;
      choice = evidence.constant;
    } else if (algorithm != nullptr &&
               algorithm->getType()->isEnumeralType() &&
               algorithm->getIdentifier() != nullptr) {
      ++matches;
      wrapperAlgorithm = algorithm;
      choice = evidence.constant;
    }
  }
  return matches == 1 && choice != nullptr;
}

class DirectWrapperEffectVisitor
    : public clang::RecursiveASTVisitor<
          DirectWrapperEffectVisitor> {
public:
  explicit DirectWrapperEffectVisitor(
      const clang::CUDAKernelCallExpr *launch,
      ASTContext &context)
      : launch(launch), context(context) {}

  bool VisitCallExpr(clang::CallExpr *call) {
    if (call != launch &&
        call != launch->getConfig())
      calls.push_back(call);
    return true;
  }

  bool VisitVarDecl(VarDecl *variable) {
    if (variable->isStaticLocal() ||
        variable->getType().isVolatileQualified() ||
        variable->hasAttr<clang::CleanupAttr>())
      safe = false;
    QualType element =
        context.getBaseElementType(variable->getType());
    const CXXRecordDecl *record =
        element.isNull()
            ? nullptr
            : element->getAsCXXRecordDecl();
    if (record != nullptr &&
        !record->hasTrivialDestructor())
      safe = false;
    return safe;
  }

  bool VisitDeclRefExpr(DeclRefExpr *reference) {
    const auto *variable =
        llvm::dyn_cast<VarDecl>(
            reference->getDecl());
    if (variable != nullptr &&
        variable->getType().isVolatileQualified())
      safe = false;
    return safe;
  }

  bool VisitImplicitCastExpr(
      clang::ImplicitCastExpr *cast) {
    if (volatileLValueToRValue(cast))
      safe = false;
    return safe;
  }

  bool VisitAtomicExpr(clang::AtomicExpr *) {
    safe = false;
    return false;
  }

  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (operation->isAssignmentOp() ||
        operation->getOpcode() == clang::BO_LAnd ||
        operation->getOpcode() == clang::BO_LOr)
      safe = false;
    return safe;
  }

  bool VisitUnaryOperator(
      clang::UnaryOperator *operation) {
    switch (operation->getOpcode()) {
    case clang::UO_PreInc:
    case clang::UO_PostInc:
    case clang::UO_PreDec:
    case clang::UO_PostDec:
      safe = false;
      break;
    default:
      break;
    }
    return safe;
  }

  bool VisitCXXConstructExpr(
      clang::CXXConstructExpr *construction) {
    const CXXRecordDecl *record =
        construction->getType()->getAsCXXRecordDecl();
    const clang::CXXConstructorDecl *constructor =
        construction->getConstructor();
    const SourceManager &sourceManager =
        context.getSourceManager();
    if (record == nullptr ||
        constructor == nullptr ||
        record->getName() != "dim3" ||
        !sourceManager.isInSystemHeader(
            record->getLocation()) ||
        !sourceManager.isInSystemHeader(
            constructor->getLocation()))
      safe = false;
    return safe;
  }

  bool VisitIfStmt(IfStmt *statement) {
    conditionals.push_back(statement);
    return true;
  }

  bool VisitForStmt(clang::ForStmt *) {
    safe = false;
    return false;
  }

  bool VisitWhileStmt(clang::WhileStmt *) {
    safe = false;
    return false;
  }

  bool VisitDoStmt(clang::DoStmt *) {
    safe = false;
    return false;
  }

  bool VisitSwitchStmt(clang::SwitchStmt *) {
    safe = false;
    return false;
  }

  bool VisitConditionalOperator(
      clang::ConditionalOperator *) {
    safe = false;
    return false;
  }

  bool VisitLambdaExpr(clang::LambdaExpr *) {
    safe = false;
    return false;
  }

  bool VisitReturnStmt(clang::ReturnStmt *statement) {
    returns.push_back(statement);
    return true;
  }

  bool VisitGotoStmt(clang::GotoStmt *) {
    safe = false;
    return false;
  }

  bool VisitIndirectGotoStmt(
      clang::IndirectGotoStmt *) {
    safe = false;
    return false;
  }

  bool VisitCXXTryStmt(clang::CXXTryStmt *) {
    safe = false;
    return false;
  }

  bool VisitGCCAsmStmt(clang::GCCAsmStmt *) {
    safe = false;
    return false;
  }

  bool VisitMSAsmStmt(clang::MSAsmStmt *) {
    safe = false;
    return false;
  }

  bool VisitCXXThrowExpr(clang::CXXThrowExpr *) {
    safe = false;
    return false;
  }

  bool VisitCXXNewExpr(clang::CXXNewExpr *) {
    safe = false;
    return false;
  }

  bool VisitCXXDeleteExpr(clang::CXXDeleteExpr *) {
    safe = false;
    return false;
  }

  const clang::CUDAKernelCallExpr *launch;
  ASTContext &context;
  bool safe = true;
  std::vector<const clang::CallExpr *> calls;
  std::vector<const IfStmt *> conditionals;
  std::vector<const clang::ReturnStmt *> returns;
};

bool allCallCandidatesNamed(
    const clang::CallExpr *call,
    const FunctionDecl *caller,
    llvm::StringRef name) {
  std::set<const FunctionDecl *> candidates;
  if (!callCandidateFunctions(
          call, caller, candidates) ||
      candidates.empty())
    return false;
  for (const FunctionDecl *candidate : candidates) {
    if (candidate->getName() != name)
      return false;
  }
  return true;
}

bool allSystemCallCandidatesNamed(
    const clang::CallExpr *call,
    const FunctionDecl *caller,
    llvm::StringRef name,
    ASTContext &context) {
  std::set<const FunctionDecl *> candidates;
  if (!callCandidateFunctions(
          call, caller, candidates) ||
      candidates.empty())
    return false;
  const SourceManager &sourceManager =
      context.getSourceManager();
  for (const FunctionDecl *candidate : candidates) {
    if (candidate->getName() != name ||
        (!sourceManager.isInSystemHeader(
             candidate->getLocation()) &&
         candidate->getBuiltinID() == 0))
      return false;
  }
  return true;
}

bool allSystemDependentCallCandidatesNamed(
    const clang::CallExpr *call,
    const FunctionDecl *caller,
    llvm::StringRef name,
    ASTContext &context) {
  if (allSystemCallCandidatesNamed(
          call, caller, name, context))
    return true;
  const auto *unresolved =
      llvm::dyn_cast_or_null<
          clang::UnresolvedLookupExpr>(
              stripExpr(
                  call == nullptr
                      ? nullptr
                      : call->getCallee()));
  if (unresolved == nullptr ||
      unresolved->requiresADL() ||
      unresolved->getName().getAsString() != name ||
      unresolved->decls_begin() ==
          unresolved->decls_end())
    return false;
  const SourceManager &sourceManager =
      context.getSourceManager();
  for (const NamedDecl *declaration :
       unresolved->decls()) {
    const FunctionDecl *candidate =
        functionFromNamedDecl(declaration);
    if (candidate == nullptr ||
        candidate->getName() != name ||
        (candidate->getBuiltinID() == 0 &&
         !sourceManager.isInSystemHeader(
             candidate->getLocation())))
      return false;
  }
  return true;
}

bool exactPreservedDependentOccupancyCall(
    const clang::CallExpr *call,
    const FunctionDecl *caller,
    ASTContext &context) {
  const auto *unresolved =
      llvm::dyn_cast_or_null<
          clang::UnresolvedLookupExpr>(
              stripExpr(
                  call == nullptr
                      ? nullptr
                      : call->getCallee()));
  static constexpr llvm::StringLiteral Name(
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
  if (unresolved == nullptr ||
      !unresolved->requiresADL() ||
      static_cast<bool>(
          unresolved->getQualifier()) ||
      unresolved->getName().getAsString() != Name ||
      unresolved->decls_begin() ==
          unresolved->decls_end() ||
      call->getNumArgs() != 4)
    return false;

  const SourceManager &sourceManager =
      context.getSourceManager();
  for (const NamedDecl *declaration :
       unresolved->decls()) {
    const FunctionDecl *candidate =
        functionFromNamedDecl(declaration);
    if (candidate == nullptr ||
        candidate->getName() != Name ||
        (candidate->getBuiltinID() == 0 &&
         !sourceManager.isInSystemHeader(
             candidate->getLocation())))
      return false;
  }

  const auto *address =
      llvm::dyn_cast_or_null<clang::UnaryOperator>(
          stripExpr(call->getArg(0)));
  const VarDecl *output =
      address != nullptr &&
              address->getOpcode() ==
                  clang::UO_AddrOf
          ? directReferencedAs<VarDecl>(
                address->getSubExpr())
          : nullptr;
  return output != nullptr &&
         !llvm::isa<ParmVarDecl>(output) &&
         output->isLocalVarDecl() &&
         !output->isStaticLocal() &&
         output->getType()->isIntegerType() &&
         systemCallHasOnlyLocalPointerEffects(
             call, context) &&
         caller != nullptr;
}

class LaunchGeometryCalleeVisitor
    : public clang::RecursiveASTVisitor<
          LaunchGeometryCalleeVisitor> {
public:
  LaunchGeometryCalleeVisitor(
      const FunctionDecl *function,
      const ParmVarDecl *output,
      bool preserveDependentOccupancy,
      ASTContext &context)
      : function(function), output(output),
        preserveDependentOccupancy(
            preserveDependentOccupancy),
        context(context) {}

  bool VisitVarDecl(VarDecl *variable) {
    if (variable->isStaticLocal() ||
        variable->getType().isVolatileQualified() ||
        variable->hasAttr<clang::CleanupAttr>())
      safe = false;
    QualType element =
        context.getBaseElementType(variable->getType());
    const CXXRecordDecl *record =
        element.isNull()
            ? nullptr
            : element->getAsCXXRecordDecl();
    if (record != nullptr &&
        !record->hasTrivialDestructor())
      safe = false;
    return safe;
  }

  bool VisitDeclRefExpr(DeclRefExpr *reference) {
    const auto *variable =
        llvm::dyn_cast<VarDecl>(
            reference->getDecl());
    if (variable != nullptr &&
        variable->hasGlobalStorage() &&
        !variable->isConstexpr() &&
        !variable->getType().isConstQualified())
      safe = false;
    return safe;
  }

  bool VisitImplicitCastExpr(
      clang::ImplicitCastExpr *cast) {
    if (volatileLValueToRValue(cast))
      safe = false;
    return safe;
  }

  bool VisitAtomicExpr(clang::AtomicExpr *) {
    safe = false;
    return false;
  }

  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (!operation->isAssignmentOp())
      return true;
    const Expr *left = stripExpr(operation->getLHS());
    const Decl *root = storageRoot(left);
    if (root == output) {
      ++outputWrites;
      outputWrite = operation;
      return true;
    }
    const VarDecl *local =
        directReferencedAs<VarDecl>(left);
    if (local == nullptr ||
        local->getDeclContext() != function)
      safe = false;
    return safe;
  }

  bool VisitUnaryOperator(
      clang::UnaryOperator *operation) {
    switch (operation->getOpcode()) {
    case clang::UO_PreInc:
    case clang::UO_PostInc:
    case clang::UO_PreDec:
    case clang::UO_PostDec:
      safe = false;
      break;
    default:
      break;
    }
    return safe;
  }

  bool VisitCallExpr(clang::CallExpr *call) {
    static const char *const allowed[] = {
        "cudaGetDevice",
        "cudaDeviceGetAttribute",
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
        "min",
        "max",
    };
    for (const char *name : allowed) {
      if (allSystemDependentCallCandidatesNamed(
              call, function, name, context) &&
          systemCallHasOnlyLocalPointerEffects(
              call, context))
        return true;
    }
    if (preserveDependentOccupancy &&
        exactPreservedDependentOccupancyCall(
            call, function, context)) {
      ++preservedDependentOccupancyCalls;
      return true;
    }
    safe = false;
    return false;
  }

  bool VisitCXXConstructExpr(
      clang::CXXConstructExpr *construction) {
    const clang::CXXConstructorDecl *constructor =
        construction->getConstructor();
    if (constructor != nullptr &&
        !constructor->isTrivial())
      safe = false;
    return safe;
  }

  bool VisitGotoStmt(clang::GotoStmt *) {
    safe = false;
    return false;
  }

  bool VisitIndirectGotoStmt(
      clang::IndirectGotoStmt *) {
    safe = false;
    return false;
  }

  bool VisitCXXTryStmt(clang::CXXTryStmt *) {
    safe = false;
    return false;
  }

  bool VisitForStmt(clang::ForStmt *) {
    safe = false;
    return false;
  }

  bool VisitWhileStmt(clang::WhileStmt *) {
    safe = false;
    return false;
  }

  bool VisitDoStmt(clang::DoStmt *) {
    safe = false;
    return false;
  }

  bool VisitSwitchStmt(clang::SwitchStmt *) {
    safe = false;
    return false;
  }

  bool VisitReturnStmt(
      clang::ReturnStmt *statement) {
    returns.push_back(statement);
    return true;
  }

  bool VisitGCCAsmStmt(clang::GCCAsmStmt *) {
    safe = false;
    return false;
  }

  bool VisitMSAsmStmt(clang::MSAsmStmt *) {
    safe = false;
    return false;
  }

  bool VisitCXXThrowExpr(clang::CXXThrowExpr *) {
    safe = false;
    return false;
  }

  bool VisitCXXNewExpr(clang::CXXNewExpr *) {
    safe = false;
    return false;
  }

  bool VisitCXXDeleteExpr(clang::CXXDeleteExpr *) {
    safe = false;
    return false;
  }

  const FunctionDecl *function;
  const ParmVarDecl *output;
  bool preserveDependentOccupancy;
  ASTContext &context;
  bool safe = true;
  unsigned preservedDependentOccupancyCalls = 0;
  unsigned outputWrites = 0;
  const BinaryOperator *outputWrite = nullptr;
  std::vector<const clang::ReturnStmt *> returns;
};

bool returnIsProvenFailure(
    const clang::ReturnStmt *returned,
    ASTContext &context) {
  const Expr *value =
      returned == nullptr
          ? nullptr
          : stripExpr(returned->getRetValue());
  Expr::EvalResult evaluated;
  if (value != nullptr &&
      value->EvaluateAsInt(evaluated, context) &&
      evaluated.Val.isInt())
    return evaluated.Val.getInt() != 0;

  const Decl *root = directReferencedDecl(value);
  if (root == nullptr)
    return false;
  const Stmt *current = returned;
  std::set<const Stmt *> active;
  while (current != nullptr &&
         active.insert(current).second) {
    const auto parents = context.getParents(*current);
    const Stmt *next = nullptr;
    for (const clang::DynTypedNode &parent :
         parents) {
      const auto *conditional =
          parent.get<IfStmt>();
      if (conditional == nullptr) {
        if (next == nullptr)
          next = parent.get<Stmt>();
        continue;
      }
      const bool exactThen =
          stripAttributedStmt(
              onlyStatement(
                  conditional->getThen())) ==
          returned;
      const bool exactElse =
          conditional->getElse() != nullptr &&
          stripAttributedStmt(
              onlyStatement(
                  conditional->getElse())) ==
              returned;
      if (!exactThen && !exactElse) {
        if (next == nullptr)
          next = conditional;
        continue;
      }
      const auto *comparison =
          llvm::dyn_cast_or_null<BinaryOperator>(
              stripExpr(conditional->getCond()));
      if (comparison == nullptr)
        continue;
      const bool nonzeroBranch =
          (exactThen &&
           comparison->getOpcode() == clang::BO_NE) ||
          (exactElse &&
           comparison->getOpcode() == clang::BO_EQ);
      if (!nonzeroBranch)
        continue;
      const Expr *other = nullptr;
      if (directReferencedDecl(
              comparison->getLHS()) == root)
        other = comparison->getRHS();
      else if (directReferencedDecl(
                   comparison->getRHS()) == root)
        other = comparison->getLHS();
      if (other != nullptr &&
          evaluatesToZero(other, context))
        return true;
    }
    current = next;
  }
  return false;
}

bool everySuccessfulCfgExitFollowsWrite(
    const FunctionDecl *function,
    const BinaryOperator *write,
    ASTContext &context) {
  if (function == nullptr || write == nullptr ||
      !function->hasBody())
    return false;
  clang::CFG::BuildOptions options;
  options.setAlwaysAdd(
      Stmt::BinaryOperatorClass);
  options.setAlwaysAdd(
      Stmt::ReturnStmtClass);
  std::unique_ptr<clang::CFG> cfg =
      clang::CFG::buildCFG(
          function,
          const_cast<Stmt *>(function->getBody()),
          &context, options);
  if (!cfg)
    return false;

  struct State {
    const clang::CFGBlock *block;
    bool followsWrite;
  };
  std::vector<State> pending = {
      {&cfg->getEntry(), false}};
  std::set<std::pair<unsigned, bool>> visited;
  bool sawReachableWrite = false;
  unsigned successfulReturns = 0;
  while (!pending.empty()) {
    const State state = pending.back();
    pending.pop_back();
    if (state.block == nullptr ||
        !visited
             .insert(
                 {state.block->getBlockID(),
                  state.followsWrite})
             .second)
      continue;

    bool followsWrite = state.followsWrite;
    bool terminated = false;
    for (const clang::CFGElement &element :
         *state.block) {
      const auto statement =
          element.getAs<clang::CFGStmt>();
      if (!statement)
        continue;
      const Stmt *node = statement->getStmt();
      if (node == write) {
        followsWrite = true;
        sawReachableWrite = true;
      }
      const auto *returned =
          llvm::dyn_cast<clang::ReturnStmt>(node);
      if (returned == nullptr)
        continue;
      terminated = true;
      if (returnIsProvenFailure(
              returned, context))
        break;
      if (!evaluatesToZero(
              returned->getRetValue(), context) ||
          !followsWrite)
        return false;
      ++successfulReturns;
      break;
    }
    if (terminated)
      continue;
    if (state.block == &cfg->getExit())
      return false;
    for (const clang::CFGBlock::AdjacentBlock
             &successor : state.block->succs()) {
      if (successor.getReachableBlock() != nullptr)
        pending.push_back(
            {successor.getReachableBlock(),
             followsWrite});
    }
  }
  return sawReachableWrite &&
         successfulReturns == 1;
}

bool exactPositiveGridWrite(
    const LaunchGeometryCalleeVisitor &effects,
    const FunctionDecl *function,
    ASTContext &context) {
  if (effects.outputWrites != 1 ||
      effects.outputWrite == nullptr ||
      !enclosingControlBranches(
           effects.outputWrite, context).empty())
    return false;
  const auto *maximum =
      llvm::dyn_cast_or_null<clang::CallExpr>(
          stripExpr(
              effects.outputWrite->getRHS()));
  if (maximum == nullptr ||
      maximum->getNumArgs() != 2 ||
      !allSystemCallCandidatesNamed(
          maximum, function, "max", context))
    return false;
  const Expr *minimumExpression = nullptr;
  if (integerValue(
          maximum->getArg(0), context, 1))
    minimumExpression = maximum->getArg(1);
  else if (integerValue(
               maximum->getArg(1), context, 1))
    minimumExpression = maximum->getArg(0);
  const auto *minimum =
      llvm::dyn_cast_or_null<clang::CallExpr>(
          stripExpr(minimumExpression));
  if (minimum == nullptr ||
      minimum->getNumArgs() != 2 ||
      !allSystemCallCandidatesNamed(
          minimum, function, "min", context))
    return false;
  return everySuccessfulCfgExitFollowsWrite(
      function, effects.outputWrite, context);
}

bool provenLaunchGeometryCall(
    const clang::CallExpr *call,
    const FunctionDecl *caller,
    const clang::CUDAKernelCallExpr *launch,
    bool preserveDependentOccupancy,
    ASTContext &context) {
  if (call == nullptr || caller == nullptr ||
      launch == nullptr ||
      launch->getConfig() == nullptr)
    return false;
  const VarDecl *grid = nullptr;
  unsigned outputIndex = 0;
  for (unsigned index = 0;
       index < call->getNumArgs(); ++index) {
    const auto *address =
        llvm::dyn_cast_or_null<
            clang::UnaryOperator>(
                stripExpr(call->getArg(index)));
    const VarDecl *candidate =
        address != nullptr &&
                address->getOpcode() ==
                    clang::UO_AddrOf
            ? directReferencedAs<VarDecl>(
                  address->getSubExpr())
            : nullptr;
    if (candidate == nullptr ||
        llvm::isa<ParmVarDecl>(candidate) ||
        !candidate->isLocalVarDecl() ||
        candidate->isStaticLocal() ||
        candidate->getType().isVolatileQualified() ||
        !candidate->getType()->isIntegerType() ||
        !containsDecl(
            launch->getConfig()->getArg(0),
            candidate))
      continue;
    if (grid != nullptr)
      return false;
    grid = candidate;
    outputIndex = index;
  }
  if (grid == nullptr)
    return false;
  std::set<const FunctionDecl *> candidates;
  if (!callCandidateFunctions(
          call, caller, candidates) ||
      candidates.empty())
    return false;
  for (const FunctionDecl *candidate : candidates) {
    const FunctionDecl *definition = nullptr;
    if (!candidate->hasBody(definition) ||
        definition == nullptr ||
        outputIndex >= definition->getNumParams())
      return false;
    if (const FunctionDecl *pattern =
            definition
                ->getTemplateInstantiationPattern()) {
      if (pattern->hasBody())
        definition = pattern;
    }
    if (outputIndex >= definition->getNumParams())
      return false;
    const ParmVarDecl *output =
        definition->getParamDecl(outputIndex);
    if (!output->getType()->isPointerType() ||
        !output->getType()
             ->getPointeeType()->isIntegerType())
      return false;
    for (unsigned index = 0;
         index < definition->getNumParams(); ++index) {
      if (index == outputIndex)
        continue;
      QualType type =
          definition->getParamDecl(index)
              ->getType();
      if (type->isReferenceType() &&
          !type.getNonReferenceType()
               .isConstQualified())
        return false;
      type = type.getNonReferenceType();
      if (type->isPointerType() &&
          !type->isFunctionPointerType())
        return false;
    }
    LaunchGeometryCalleeVisitor effects(
        definition, output,
        preserveDependentOccupancy, context);
    effects.TraverseStmt(
        const_cast<Stmt *>(definition->getBody()));
    if (preserveDependentOccupancy &&
        effects.preservedDependentOccupancyCalls != 1)
      return false;
    const bool positive =
        exactPositiveGridWrite(
            effects, definition, context);
    if (!effects.safe || !positive)
      return false;
  }
  return true;
}

const VarDecl *enclosingInitializedVariable(
    const Expr *expression,
    ASTContext &context) {
  const Stmt *current = expression;
  std::set<const Stmt *> active;
  while (current != nullptr &&
         active.insert(current).second) {
    const auto parents = context.getParents(*current);
    const Stmt *next = nullptr;
    for (const clang::DynTypedNode &parent : parents) {
      if (const auto *variable =
              parent.get<VarDecl>()) {
        return variable->hasInit() ? variable : nullptr;
      }
      if (next == nullptr)
        next = parent.get<Stmt>();
    }
    current = next;
  }
  return nullptr;
}

bool directWrapperEffectsExact(
    const FunctionDecl *function,
    const clang::CUDAKernelCallExpr *launch,
    bool preserveGeometry,
    const clang::CompoundStmt *&postGeometryBlock,
    ASTContext &context) {
  postGeometryBlock = nullptr;
  DirectWrapperEffectVisitor effects(
      launch, context);
  effects.TraverseStmt(
      const_cast<Stmt *>(function->getBody()));
  if (!effects.safe ||
      effects.calls.size() != 2 ||
      effects.conditionals.size() != 1 ||
      effects.returns.size() != 2)
    return false;
  const clang::CallExpr *setup = nullptr;
  const clang::CallExpr *peek = nullptr;
  for (const clang::CallExpr *call : effects.calls) {
    const bool geometry =
        provenLaunchGeometryCall(
            call, function, launch,
            preserveGeometry, context);
    const bool peekCall =
        allSystemCallCandidatesNamed(
            call, function,
            "cudaPeekAtLastError",
            context);
    if (geometry) {
      if (setup != nullptr)
        return false;
      setup = call;
    } else if (peekCall) {
      if (peek != nullptr)
        return false;
      peek = call;
    } else {
      return false;
    }
  }
  if (setup == nullptr || peek == nullptr)
    return false;
  const VarDecl *error =
      enclosingInitializedVariable(setup, context);
  if (error == nullptr ||
      error->getType().isVolatileQualified() ||
      stripExpr(error->getInit()) != setup ||
      !context.hasSameType(
          error->getType().getCanonicalType(),
          function->getReturnType()
              .getCanonicalType()))
    return false;
  const IfStmt *errorCheck =
      effects.conditionals.front();
  const clang::CompoundStmt *geometryBlock =
      nullptr;
  for (const clang::DynTypedNode &parent :
       context.getParents(*errorCheck)) {
    if (const auto *compound =
            parent.get<clang::CompoundStmt>()) {
      if (geometryBlock != nullptr)
        return false;
      geometryBlock = compound;
    }
  }
  const auto *errorBody =
      llvm::dyn_cast_or_null<clang::CompoundStmt>(
          stripAttributedStmt(
              errorCheck->getThen()));
  const Stmt *errorReturn =
      onlyStatement(errorCheck->getThen());
  const auto *returnStatement =
      llvm::dyn_cast_or_null<clang::ReturnStmt>(
          stripAttributedStmt(errorReturn));
  const auto *comparison =
      llvm::dyn_cast_or_null<BinaryOperator>(
          stripExpr(errorCheck->getCond()));
  bool exactErrorCheck = false;
  if (comparison != nullptr &&
      comparison->getOpcode() == clang::BO_NE) {
    const Expr *other = nullptr;
    if (directReferencedAs<VarDecl>(
            comparison->getLHS()) == error)
      other = comparison->getRHS();
    else if (directReferencedAs<VarDecl>(
                 comparison->getRHS()) == error)
      other = comparison->getLHS();
    exactErrorCheck =
        other != nullptr &&
        evaluatesToZero(other, context);
  }
  const clang::DeclStmt *errorDeclaration =
      nullptr;
  bool exactGeometryBlock = false;
  if (geometryBlock != nullptr &&
      geometryBlock->size() == 2) {
    auto statement = geometryBlock->body_begin();
    errorDeclaration =
        llvm::dyn_cast_or_null<clang::DeclStmt>(
            stripAttributedStmt(*statement));
    ++statement;
    exactGeometryBlock =
        stripAttributedStmt(*statement) ==
            errorCheck &&
        errorDeclaration != nullptr &&
        errorDeclaration->isSingleDecl() &&
        errorDeclaration->getSingleDecl() == error;
  }
  if (errorCheck->getElse() != nullptr ||
      errorBody == nullptr ||
      !exactGeometryBlock ||
      !exactErrorCheck ||
      returnStatement == nullptr ||
      directReferencedAs<VarDecl>(
          returnStatement->getRetValue()) != error)
    return false;
  unsigned peekReturns = 0;
  for (const clang::ReturnStmt *statement :
       effects.returns) {
    if (stripExpr(statement->getRetValue()) == peek)
      ++peekReturns;
  }
  const SourceManager &sourceManager =
      context.getSourceManager();
  const SourceLocation setupLocation =
      sourceManager.getExpansionLoc(
          setup->getBeginLoc());
  const SourceLocation checkLocation =
      sourceManager.getExpansionLoc(
          errorCheck->getBeginLoc());
  const SourceLocation geometryEndLocation =
      sourceManager.getExpansionLoc(
          geometryBlock->getEndLoc());
  const SourceLocation launchLocation =
      sourceManager.getExpansionLoc(
          launch->getBeginLoc());
  const SourceLocation peekLocation =
      sourceManager.getExpansionLoc(
          peek->getBeginLoc());
  const bool exactOrder =
      peekReturns == 1 &&
      sourceManager.isBeforeInTranslationUnit(
             setupLocation, checkLocation) &&
      sourceManager.isBeforeInTranslationUnit(
             geometryEndLocation,
             launchLocation) &&
      sourceManager.isBeforeInTranslationUnit(
             launchLocation, peekLocation);
  if (exactOrder && preserveGeometry)
    postGeometryBlock = geometryBlock;
  return exactOrder;
}

DirectLaunchWrapperProof proveDirectLaunchWrapper(
    const FunctionDecl *function,
    ASTContext &context,
    SourceManager &sourceManager,
    const std::set<const FunctionDecl *> &softmaxKernels,
    const SoftmaxKernelBindings &softmaxBindings,
    const std::set<const FunctionDecl *> &rmsKernels) {
  DirectLaunchWrapperProof proof;
  if (function == nullptr || !function->hasBody() ||
      llvm::isa<clang::CXXMethodDecl>(function) ||
      function->hasAttr<clang::CUDADeviceAttr>() ||
      function->hasAttr<clang::CUDAGlobalAttr>() ||
      !statusReturnCompatible(function, sourceManager))
    return proof;

  ExecutableCallCollector collector;
  collector.context = &context;
  collector.TraverseStmt(
      const_cast<Stmt *>(function->getBody()));
  const clang::CUDAKernelCallExpr *launch = nullptr;
  unsigned launches = 0;
  for (const ExecutableCallCollector::FoundCall &found :
       collector.calls) {
    const auto *candidate =
        llvm::dyn_cast<clang::CUDAKernelCallExpr>(
            found.call);
    if (candidate == nullptr)
      continue;
    ++launches;
    if (found.conditionalDepth != 0)
      return proof;
    launch = candidate;
  }
  if (launches != 1 || launch == nullptr)
    return proof;

  std::set<const FunctionDecl *> candidates;
  if (!callCandidateFunctions(
          launch, function, candidates) ||
      candidates.size() != 1)
    return proof;
  const FunctionDecl *kernel =
      *candidates.begin();
  const bool softmax =
      softmaxKernels.count(
          kernel->getCanonicalDecl()) != 0;
  const bool rms =
      rmsKernels.count(
          kernel->getCanonicalDecl()) != 0;
  if (softmax == rms)
    return proof;
  const clang::CompoundStmt *postGeometryBlock =
      nullptr;
  if (!directWrapperEffectsExact(
          function, launch, rms,
          postGeometryBlock, context))
    return proof;

  const TemplateTypeParmDecl *loadType = nullptr;
  const TemplateTypeParmDecl *storeType = nullptr;
  const TemplateTypeParmDecl *computeType = nullptr;
  std::vector<clang::TemplateArgument> arguments;
  if (!wrapperTemplateMapping(
          function, launch, kernel,
          loadType, storeType, computeType,
          arguments))
    return proof;

  const unsigned expectedKernelArguments =
      softmax ? 4 : 6;
  if (launch->getNumArgs() !=
      expectedKernelArguments)
    return proof;
  bool ignoredCast = false;
  const ParmVarDecl *load =
      parameterThroughIntegralCasts(
          launch->getArg(0), ignoredCast);
  if (ignoredCast)
    return proof;
  const ParmVarDecl *store =
      parameterThroughIntegralCasts(
          launch->getArg(1), ignoredCast);
  if (ignoredCast ||
      !typeIsTemplateParameter(
          load == nullptr ? QualType() : load->getType(),
          loadType, false) ||
      !typeIsTemplateParameter(
          store == nullptr ? QualType() : store->getType(),
          storeType, false))
    return proof;
  bool rowsCast = false;
  bool columnsCast = false;
  const ParmVarDecl *rows =
      parameterThroughIntegralCasts(
          launch->getArg(2), rowsCast);
  const ParmVarDecl *columns =
      parameterThroughIntegralCasts(
          launch->getArg(3), columnsCast);
  if (rows == nullptr || columns == nullptr ||
      rows == columns ||
      (softmax && (rowsCast || columnsCast)) ||
      !isIntegralWidthAtLeast(
          rows->getType(), context, 32) ||
      !isIntegralWidthAtLeast(
          columns->getType(), context, 32))
    return proof;

  const auto *configuration = launch->getConfig();
  if (configuration == nullptr ||
      configuration->getNumArgs() != 4)
    return proof;
  const ParmVarDecl *stream =
      parameterThroughIntegralCasts(
          configuration->getArg(3), ignoredCast);
  if (stream == nullptr || ignoredCast ||
      !stream->getType()->isPointerType())
    return proof;
  const ParmVarDecl *sharedMemory =
      parameterThroughIntegralCasts(
          configuration->getArg(2), ignoredCast);
  if (ignoredCast)
    return proof;
  if (sharedMemory == nullptr) {
    uint64_t sharedMemoryValue = 0;
    if (!evaluatesToPositiveInteger(
            configuration->getArg(2),
            context, sharedMemoryValue)) {
      Expr::EvalResult zero;
      if (!configuration->getArg(2)->EvaluateAsInt(
              zero, context) ||
          !zero.Val.isInt() ||
          zero.Val.getInt() != 0)
        return proof;
    }
  } else if (!isIntegralWidthAtLeast(
                 sharedMemory->getType(),
                 context, 32)) {
    return proof;
  }

  const ParmVarDecl *epsilon = nullptr;
  const ParmVarDecl *inverseOutput = nullptr;
  if (!softmax) {
    epsilon = parameterThroughIntegralCasts(
        launch->getArg(4), ignoredCast);
    if (epsilon == nullptr || ignoredCast ||
        !epsilon->getType()
             .getNonReferenceType()->isFloatingType())
      return proof;
    inverseOutput = parameterThroughIntegralCasts(
        launch->getArg(5), ignoredCast);
    if (inverseOutput == nullptr || ignoredCast ||
        !typeIsTemplateParameter(
            inverseOutput->getType(),
            computeType, true))
      return proof;
  }

  std::set<const ParmVarDecl *> expected = {
      stream, load, store, rows, columns};
  if (epsilon != nullptr)
    expected.insert(epsilon);
  if (inverseOutput != nullptr)
    expected.insert(inverseOutput);
  if (sharedMemory != nullptr)
    expected.insert(sharedMemory);
  if (expected.size() != function->getNumParams())
    return proof;
  for (const ParmVarDecl *parameter :
       function->parameters()) {
    if (parameter->getIdentifier() == nullptr ||
        expected.count(parameter) == 0)
      return proof;
  }

  const NonTypeTemplateParmDecl *algorithm = nullptr;
  const EnumConstantDecl *choice = nullptr;
  if (softmax &&
      !softmaxLaunchChoice(
          kernel, arguments, softmaxBindings,
          algorithm, choice))
    return proof;
  if (!directKernelDomainMapping(
          function, kernel, arguments,
          softmax, proof))
    return proof;

  proof.softmax = softmax;
  proof.function = function;
  proof.stream = stream;
  proof.load = load;
  proof.store = store;
  proof.rows = rows;
  proof.columns = columns;
  proof.epsilon = epsilon;
  proof.inverseOutput = inverseOutput;
  proof.sharedMemory = sharedMemory;
  proof.computeType = computeType;
  proof.algorithm = algorithm;
  proof.choice = choice;
  proof.postGeometryBlock =
      postGeometryBlock;
  proof.rowsIntegralCast = rowsCast;
  proof.columnsIntegralCast = columnsCast;
  return proof;
}

const FunctionDecl *asFunctionDecl(const NamedDecl *declaration) {
  if (const auto *function =
          llvm::dyn_cast_or_null<FunctionDecl>(declaration))
    return function;
  if (const auto *functionTemplate =
          llvm::dyn_cast_or_null<FunctionTemplateDecl>(declaration))
    return functionTemplate->getTemplatedDecl();
  return nullptr;
}

class ReferencedFunctionVisitor
    : public clang::RecursiveASTVisitor<ReferencedFunctionVisitor> {
public:
  bool VisitCallExpr(clang::CallExpr *call) {
    if (const FunctionDecl *direct = call->getDirectCallee())
      functions.insert(direct->getCanonicalDecl());
    const Expr *callee = stripExpr(call->getCallee());
    if (const auto *reference =
            llvm::dyn_cast_or_null<DeclRefExpr>(callee)) {
      if (const FunctionDecl *function =
              asFunctionDecl(reference->getDecl()))
        functions.insert(function->getCanonicalDecl());
    }
    if (const auto *unresolved =
            llvm::dyn_cast_or_null<clang::UnresolvedLookupExpr>(
                callee)) {
      for (const NamedDecl *declaration : unresolved->decls()) {
        if (const FunctionDecl *function =
                asFunctionDecl(declaration))
          functions.insert(function->getCanonicalDecl());
      }
    }
    return true;
  }

  std::set<const FunctionDecl *> functions;
};

bool isDirectColumnThreshold(const Expr *condition,
                             const ParmVarDecl *columns,
                             ASTContext &context) {
  const auto *comparison =
      llvm::dyn_cast_or_null<BinaryOperator>(stripExpr(condition));
  if (comparison == nullptr ||
      (comparison->getOpcode() != clang::BO_LT &&
       comparison->getOpcode() != clang::BO_LE) ||
      directReferencedAs<ParmVarDecl>(
          comparison->getLHS()) != columns)
    return false;
  uint64_t threshold = 0;
  return evaluatesToPositiveInteger(
             comparison->getRHS(), context, threshold) &&
         threshold >= 32 && threshold <= 65536;
}

class BoolLocalVisitor
    : public clang::RecursiveASTVisitor<BoolLocalVisitor> {
public:
  bool VisitVarDecl(VarDecl *variable) {
    if (variable->getType()
            .getNonReferenceType()
            .getUnqualifiedType()->isBooleanType())
      found = true;
    return !found;
  }

  bool contains(const Stmt *statement) {
    found = false;
    TraverseStmt(const_cast<Stmt *>(statement));
    return found;
  }

private:
  bool found = false;
};

bool hasRowWidthRoutingShape(const FunctionDecl *function,
                             ASTContext &context,
                             unsigned widthParameterIndex,
                             unsigned forwardedParameterCount) {
  const auto *body =
      llvm::dyn_cast_or_null<clang::CompoundStmt>(
          function->getBody());
  if (body == nullptr || body->size() != 1 ||
      widthParameterIndex >= function->getNumParams() ||
      forwardedParameterCount > function->getNumParams())
    return false;
  const auto *route = llvm::dyn_cast_or_null<IfStmt>(
      stripAttributedStmt(*body->body_begin()));
  if (route == nullptr || route->getElse() == nullptr ||
      !isDirectColumnThreshold(
          route->getCond(),
          function->getParamDecl(widthParameterIndex),
          context))
    return false;

  ExecutableCallCollector smallRoute;
  smallRoute.context = &context;
  smallRoute.TraverseStmt(
      const_cast<Stmt *>(route->getThen()));
  unsigned smallForwarding = 0;
  for (const ExecutableCallCollector::FoundCall &found :
       smallRoute.calls) {
    if (callForwardsParameters(
            found.call, function,
            forwardedParameterCount))
      ++smallForwarding;
  }
  if (smallForwarding == 0)
    return false;

  ExecutableCallCollector largeRoute;
  largeRoute.context = &context;
  largeRoute.TraverseStmt(
      const_cast<Stmt *>(route->getElse()));
  unsigned largeForwarding = 0;
  for (const ExecutableCallCollector::FoundCall &found :
       largeRoute.calls) {
    if (callForwardsParameters(
            found.call, function,
            forwardedParameterCount))
      ++largeForwarding;
  }
  BoolLocalVisitor boolLocal;
  return largeForwarding >= 2 &&
         boolLocal.contains(route->getElse());
}

bool isRmsDispatcher(
    const FunctionDecl *function,
    ASTContext &context,
    SourceManager &sourceManager,
    const std::set<const FunctionDecl *> &rmsKernels,
    const FunctionGraph &graph) {
  if (function == nullptr || !function->hasBody() ||
      function->hasAttr<clang::CUDADeviceAttr>() ||
      function->hasAttr<clang::CUDAGlobalAttr>() ||
      !statusReturnCompatible(function, sourceManager) ||
      function->getNumParams() != 7 ||
      !exactlyTypeTemplateParameters(function, 3) ||
      !function->getParamDecl(0)->getType()->isPointerType() ||
      !templateTypeParameter(
          function->getParamDecl(1)->getType(),
          0, 0, false, false, false) ||
      !templateTypeParameter(
          function->getParamDecl(2)->getType(),
          0, 1, false, false, false) ||
      !isIntegralWidthAtLeast(
          function->getParamDecl(3)->getType(), context, 32) ||
      !isIntegralWidthAtLeast(
          function->getParamDecl(4)->getType(), context, 32) ||
      !function->getParamDecl(5)->getType()
           .getNonReferenceType()->isFloatingType() ||
      !templateTypeParameter(
          function->getParamDecl(6)->getType(),
          0, 2, true, false, true) ||
      !hasRowWidthRoutingShape(function, context, 4, 7))
    return false;

  return allForwardingRoutesReachSemanticKernels(
      function, 7, rmsKernels, graph, nullptr,
      nullptr,
      context);
}

bool validMainFileInsertion(SourceLocation location,
                            SourceManager &sourceManager) {
  return location.isValid() && !location.isMacroID() &&
         sourceManager.isWrittenInMainFile(location);
}

std::string uniqueResultName(const FunctionDecl *function) {
  static const char *base = "ascify_dav_c310_recipe_result";
  class NameVisitor : public clang::RecursiveASTVisitor<NameVisitor> {
  public:
    bool VisitNamedDecl(NamedDecl *declaration) {
      if (declaration->getIdentifier() != nullptr)
        names.insert(
            declaration->getNameAsString());
      return true;
    }
    std::set<std::string> names;
  };
  NameVisitor visitor;
  if (const FunctionTemplateDecl *functionTemplate =
          primaryFunctionTemplate(function)) {
    const TemplateParameterList *parameters =
        functionTemplate->getTemplateParameters();
    for (unsigned index = 0;
         index < parameters->size(); ++index)
      visitor.TraverseDecl(
          const_cast<NamedDecl *>(
              parameters->getParam(index)));
  }
  for (const ParmVarDecl *parameter :
       function->parameters())
    visitor.TraverseDecl(
        const_cast<ParmVarDecl *>(parameter));
  visitor.TraverseStmt(
      const_cast<Stmt *>(function->getBody()));
  std::string candidate = base;
  unsigned suffix = 0;
  while (visitor.names.count(candidate) != 0) {
    ++suffix;
    candidate =
        std::string(base) + "_generated";
    if (suffix > 1)
      candidate += "_" + std::to_string(suffix);
  }
  return candidate;
}

std::string directWrapperPrologue(
    const DirectLaunchWrapperProof &proof) {
  const std::string result =
      uniqueResultName(proof.function);
  const std::string rows =
      proof.rows->getNameAsString();
  const std::string columns =
      proof.columns->getNameAsString();
  const std::string pack =
      proof.packSize->getNameAsString();
  const std::string macroPush =
      "\n#pragma push_macro(\"" + result +
      "\")\n#undef " + result + "\n";
  const std::string macroPop =
      "#pragma pop_macro(\"" + result +
      "\")\n";
  std::string condition =
      "(" + rows + " > 0) && (" + columns +
      " > 0) && (" + columns + " % " + pack +
      " == 0)";
  if (proof.domain ==
      DirectKernelDomain::SoftmaxWarp) {
    const std::string capacity =
        "(" +
        proof.maxColumnsPerThread->getNameAsString() +
        " * " +
        proof.threadGroupWidth->getNameAsString() +
        ")";
    condition += " && (" + columns + " <= " +
                 capacity + ")";
    condition += " && (" +
                 proof.padding->getNameAsString() +
                 " || " + columns + " == " +
                 capacity + ")";
    condition += " && (" + rows + " % " +
                 proof.rowsPerAccess->getNameAsString() +
                 " == 0)";
  } else if (proof.domain ==
             DirectKernelDomain::RmsNormWarp) {
    const std::string maximum =
        "(" +
        proof.maxColumnsPerThread->getNameAsString() +
        " * " +
        proof.threadGroupWidth->getNameAsString() +
        ")";
    const std::string minimum =
        "(" +
        proof.minColumnsPerThread->getNameAsString() +
        " * " +
        proof.threadGroupWidth->getNameAsString() +
        ")";
    condition += " && (" + columns + " <= " +
                 maximum + ")";
    condition += " && (" +
                 proof.padding->getNameAsString() +
                 " ? " + columns + " > " + minimum +
                 " : " + columns + " == " + maximum +
                 ")";
  }
  if (proof.sharedMemory != nullptr) {
    const std::string shared =
        proof.sharedMemory->getNameAsString();
    condition += " && (" + shared + " >= 0)";
    condition +=
        " && (static_cast<unsigned long long>(" +
        columns +
        ") <= static_cast<unsigned long long>(" +
        shared + ") / sizeof(" +
        proof.computeType->getNameAsString() + "))";
  }
  std::string text =
      macroPush + "  if (" + condition + ") {\n";
  if (proof.softmax && proof.algorithm != nullptr) {
    text += "    if constexpr (";
    text += proof.algorithm->getNameAsString();
    text += " == ::";
    text += proof.choice->getQualifiedNameAsString();
    text += ") {\n";
  }
  text +=
      proof.softmax && proof.algorithm != nullptr
          ? "      const auto "
          : "    const auto ";
  text += result;
  text += " = ::ascify::target::dav_c310::";
  text += proof.softmax ? "TrySoftmax(" : "TryRmsNorm(";
  text += proof.stream->getNameAsString();
  text += ", ";
  text += proof.load->getNameAsString();
  text += ", ";
  text += proof.store->getNameAsString();
  text += ", ";
  text += proof.rows->getNameAsString();
  text += ", ";
  text += proof.columns->getNameAsString();
  if (!proof.softmax) {
    text += ", ";
    text += proof.epsilon->getNameAsString();
    text += ", ";
    text += proof.inverseOutput->getNameAsString();
  }
  text += ", static_cast<";
  text += proof.computeType->getNameAsString();
  text += "*>(nullptr));\n"
          "    if (";
  text += result;
  text += ".handled) { return ";
  text += result;
  text += ".status; }\n";
  if (proof.softmax && proof.algorithm != nullptr)
    text += "    }\n";
  text += "  }\n";
  text += macroPop;
  return text;
}

SourceLocation afterLeftBrace(const FunctionDecl *function,
                              SourceManager &sourceManager,
                              const LangOptions &langOptions) {
  const auto *body =
      llvm::dyn_cast_or_null<clang::CompoundStmt>(
          function->getBody());
  if (body == nullptr)
    return SourceLocation();
  const SourceLocation rawBrace =
      body->getLBracLoc();
  if (rawBrace.isInvalid() ||
      rawBrace.isMacroID())
    return SourceLocation();
  const SourceLocation brace =
      sourceManager.getFileLoc(rawBrace);
  if (!validMainFileInsertion(brace, sourceManager))
    return SourceLocation();
  return clang::Lexer::getLocForEndOfToken(
      brace, 0, sourceManager, langOptions);
}

SourceLocation afterStatement(
    const Stmt *statement,
    SourceManager &sourceManager,
    const LangOptions &langOptions) {
  if (statement == nullptr)
    return SourceLocation();
  const SourceLocation rawEnd =
      statement->getEndLoc();
  if (rawEnd.isInvalid() || rawEnd.isMacroID())
    return SourceLocation();
  const SourceLocation end =
      sourceManager.getFileLoc(rawEnd);
  if (!validMainFileInsertion(end, sourceManager))
    return SourceLocation();
  return clang::Lexer::getLocForEndOfToken(
      end, 0, sourceManager, langOptions);
}

} // namespace

void DavC310TargetRecipe::reset() {
  functions.clear();
  records.clear();
}

void DavC310TargetRecipe::registerMatchers(
    clang::ast_matchers::MatchFinder &finder,
    clang::ast_matchers::MatchFinder::MatchCallback *callback) const {
  namespace mat = clang::ast_matchers;
  finder.addMatcher(
      mat::functionDecl(
          mat::isDefinition(),
          mat::unless(mat::isExpansionInSystemHeader()))
          .bind(FunctionBinding),
      callback);
  finder.addMatcher(
      mat::cxxRecordDecl(
          mat::isDefinition(),
          mat::isExpansionInMainFile())
          .bind(RecordBinding),
      callback);
}

bool DavC310TargetRecipe::collect(
    const clang::ast_matchers::MatchFinder::MatchResult &result) {
  if (const auto *function =
          result.Nodes.getNodeAs<FunctionDecl>(FunctionBinding)) {
    if (std::find(functions.begin(), functions.end(), function) ==
        functions.end())
      functions.push_back(function);
    return true;
  }
  if (const auto *record =
          result.Nodes.getNodeAs<CXXRecordDecl>(RecordBinding)) {
    if (std::find(records.begin(), records.end(), record) ==
        records.end())
      records.push_back(record);
    return true;
  }
  return false;
}

DavC310TargetRecipe::FinalizedEdits
DavC310TargetRecipe::finalize(
    ASTContext &context,
    SourceManager &sourceManager,
    const LangOptions &langOptions) {
  FinalizedEdits finalized;
  std::set<unsigned> insertionOffsets;

  for (const CXXRecordDecl *record : records) {
    const AdapterProof proof =
        proveDirectAdapter(record, context, sourceManager);
    if (proof.kind == AdapterKind::None)
      continue;
    const SourceLocation rawCloseBrace =
        record->getBraceRange().getEnd();
    if (rawCloseBrace.isInvalid() ||
        rawCloseBrace.isMacroID())
      continue;
    const SourceLocation closeBrace =
        sourceManager.getFileLoc(rawCloseBrace);
    if (!validMainFileInsertion(closeBrace, sourceManager))
      continue;
    const unsigned offset =
        sourceManager.getFileOffset(closeBrace);
    if (!insertionOffsets.insert(offset).second)
      continue;
    finalized.edits.push_back(
        {closeBrace, adapterMarkerText(proof),
         Edit::Kind::AdapterMarker});
    if (proof.kind == AdapterKind::RowMajorInput)
      ++finalized.directLoadAdapters;
    else
      ++finalized.directStoreAdapters;
  }

  std::set<const EnumConstantDecl *> softmaxChoices;
  std::set<const FunctionDecl *> softmaxKernels;
  SoftmaxKernelBindings softmaxBindings;
  std::set<const FunctionDecl *> rmsKernels;
  PrimitiveRegistry primitives(
      context, sourceManager, functions);
  for (const FunctionDecl *function : functions) {
    if (function == nullptr || !function->hasBody())
      continue;
    collectSoftmaxChoices(
        function, context, primitives,
        sourceManager, softmaxChoices,
        softmaxKernels, softmaxBindings);
    if (isRmsKernel(
            function, context, primitives, sourceManager))
      rmsKernels.insert(function->getCanonicalDecl());
  }
  for (const FunctionDecl *function : functions) {
    const DirectLaunchWrapperProof wrapper =
        proveDirectLaunchWrapper(
            function, context, sourceManager,
            softmaxKernels, softmaxBindings,
            rmsKernels);
    if (!wrapper.proven())
      continue;

    const SourceLocation insert =
        wrapper.softmax
            ? afterLeftBrace(
                  function, sourceManager, langOptions)
            : afterStatement(
                  wrapper.postGeometryBlock,
                  sourceManager, langOptions);
    if (!validMainFileInsertion(insert, sourceManager))
      continue;
    const unsigned offset =
        sourceManager.getFileOffset(insert);
    if (!insertionOffsets.insert(offset).second)
      continue;

    finalized.edits.push_back(
        {insert, directWrapperPrologue(wrapper),
         wrapper.softmax
             ? Edit::Kind::SoftmaxDirectWrapperPrologue
             : Edit::Kind::RmsNormDirectWrapperPrologue});
    if (wrapper.softmax)
      ++finalized.softmaxDirectWrappers;
    else
      ++finalized.rmsNormDirectWrappers;
    finalized.needsTargetHeader = true;
  }

  finalized.needsTargetHeader =
      finalized.softmaxDirectWrappers != 0 ||
      finalized.rmsNormDirectWrappers != 0;
  return finalized;
}

} // namespace ascify
